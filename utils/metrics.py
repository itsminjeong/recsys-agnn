# utils/metrics.py
# ------------------------------------------------------------
# 목적
# - Warm: RMSE/MAE + NDCG@10
# - Cold: RMSE/MAE(각 run_*에서 동일 함수 재사용) + Recall@10 + NDCG@10
#
# 설계 포인트
# - MF/GCN: model(u, i) 또는 model.predict(u, i, ...) 지원
# - MF+Attr: (users, items) -> scores 형태의 scorer 콜러블을 넘기면 됨
# - GCN 캐시(h_user, h_item)가 있으면 gcn_cache=(h_user, h_item)로 전달
# - 후보군 = 전체 아이템 - (train에서 유저가 본 아이템)  +  (정답 아이템 강제 포함)
# - negative stride / shape mismatch 이슈 방지: .contiguous(), .tolist() 등 적용
# ------------------------------------------------------------

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple, Union, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =========================
# 0) 내부 유틸
# =========================
def _build_pos_dict(df, ucol: str = "user_id", icol: str = "item_id") -> Dict[int, set]:
    """DataFrame -> user -> set(items) 매핑 생성."""
    pos: Dict[int, set] = {}
    if df is None or len(df) == 0:
        return pos
    for u, i in zip(df[ucol].values, df[icol].values):
        pos.setdefault(int(u), set()).add(int(i))
    return pos


def _make_scorer(
    model: Optional[nn.Module],
    gcn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    scorer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    (users, items) -> scores 로 점수 반환하는 함수 생성.
    우선순위: 사용자 제공 scorer > model.predict > model.forward
    """
    if scorer is not None:
        return scorer

    def _sc(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if model is None:
            raise ValueError("scorer is None and model is None. One of them must be provided.")
        if hasattr(model, "predict"):
            if gcn_cache is not None:
                hu, hi = gcn_cache
                return model.predict(users, items, hu, hi)
            return model.predict(users, items)
        return model(users, items)

    return _sc


# =========================
# 1) 회귀 지표 (Warm/Cold 공용)
# =========================
@torch.no_grad()
def eval_rmse_mae(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    predict_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    """
    DataLoader 배치가 (u, i, r) 형태라고 가정.
    - MF/GCN: model(u,i) 또는 model.predict(u,i)
    - MF+Attr: run_* 쪽에서 predict_fn(u,i)로 감싸서 전달
    """
    model.eval()

    preds, trues = [], []
    for batch in loader:
        if len(batch) < 3:
            raise ValueError("eval_rmse_mae expects batches like (u, i, r).")
        u, i, r = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        if predict_fn is not None:
            p = predict_fn(u, i)
        else:
            if hasattr(model, "predict"):
                p = model.predict(u, i)
            else:
                p = model(u, i)

        preds.append(p.detach().cpu().reshape(-1).numpy())
        trues.append(r.detach().cpu().reshape(-1).numpy())

    preds = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    trues = np.concatenate(trues) if trues else np.array([], dtype=np.float32)
    if preds.size == 0:
        return 0.0, 0.0

    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    mae  = float(np.abs(preds - trues).mean())
    return rmse, mae


# =========================
# 2) Warm NDCG@K
# =========================
@torch.no_grad()
def eval_warm_ndcg_at_k(
    model_or_scorer: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    train_df, valid_df,
    n_items: int,
    K: int = 10,
    device: str = "cpu",
    gcn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    scorer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    batch_items: int = 4096,
) -> float:
    """
    Warm 환경의 랭킹 품질(NDCG@K).
    - 후보군: 전체 아이템 - (train에서 유저가 본 아이템)  +  (정답 아이템 강제 포함)
    - 정답: valid_df 내 (user,item) 집합
    """
    # scorer 준비
    if callable(model_or_scorer) and not isinstance(model_or_scorer, nn.Module) and scorer is None:
        scorer_fn = model_or_scorer
        model = None
    else:
        model = model_or_scorer if isinstance(model_or_scorer, nn.Module) else None
        scorer_fn = _make_scorer(model, gcn_cache, scorer)

    train_pos = _build_pos_dict(train_df)
    valid_pos = _build_pos_dict(valid_df)
    users = list(valid_pos.keys())
    if len(users) == 0:
        return 0.0

    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    ndcg_sum, n_users = 0.0, 0
    for u in users:
        gt = valid_pos.get(u, set())
        if not gt:
            continue

        # 후보군 구성: train에서 본 것 제거
        u_train = train_pos.get(u, set())
        if u_train:
            mask = torch.ones(n_items, dtype=torch.bool, device=device)
            idx = torch.tensor(list(u_train), dtype=torch.long, device=device)
            mask[idx] = False
            candidates = all_items[mask]
        else:
            candidates = all_items

        # ★ 정답(gt)을 후보군에 반드시 포함
        gt_tensor = torch.tensor(sorted(gt), dtype=torch.long, device=device)
        candidates = torch.unique(torch.cat([candidates, gt_tensor]))

        # 점수 계산(배치)
        scores_list = []
        num_c = candidates.numel()
        for s in range(0, num_c, batch_items):
            e = min(s + batch_items, num_c)
            items_chunk = candidates[s:e]
            users_chunk = torch.full((items_chunk.numel(),), u, dtype=torch.long, device=device)
            sc = scorer_fn(users_chunk, items_chunk).reshape(-1).to(torch.float32)
            scores_list.append(sc)

        if not scores_list:
            continue
        scores = torch.cat(scores_list, dim=0)

        K_eff = min(K, scores.numel())
        if K_eff <= 0:
            continue

        _, topk_rel_idx = torch.topk(scores, k=K_eff, largest=True, sorted=True)
        topk_items = candidates[topk_rel_idx].contiguous().to("cpu").tolist()

        # NDCG@K
        dcg = 0.0
        for rank, it in enumerate(topk_items):
            if it in gt:
                dcg += 1.0 / math.log2(rank + 2.0)
        idcg = sum(1.0 / math.log2(r + 2.0) for r in range(min(K_eff, len(gt))))
        ndcg_u = (dcg / idcg) if idcg > 0 else 0.0

        ndcg_sum += ndcg_u
        n_users += 1

    return float(ndcg_sum / n_users) if n_users > 0 else 0.0


# =========================
# 3) Cold Recall@K / NDCG@K
# =========================
@torch.no_grad()
def eval_cold_recall_ndcg_at_k(
    model_or_scorer: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    train_df, cold_df,
    n_items: int,
    K: int = 10,
    device: str = "cpu",
    gcn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    scorer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    users_subset: Optional[Iterable[int]] = None,
    batch_items: int = 4096,
) -> Tuple[float, float]:
    """
    Cold 환경의 랭킹 품질(Recall@K, NDCG@K).
    - 후보군: 전체 아이템 - (train에서 유저가 본 아이템)  +  (정답 아이템 강제 포함)
    - 정답: cold_df 내 (user,item) 집합
    - users_subset가 주어지면 해당 유저만 평가(예: cold-user만)
    """
    # scorer 준비
    if callable(model_or_scorer) and not isinstance(model_or_scorer, nn.Module) and scorer is None:
        scorer_fn = model_or_scorer
        model = None
    else:
        model = model_or_scorer if isinstance(model_or_scorer, nn.Module) else None
        scorer_fn = _make_scorer(model, gcn_cache, scorer)

    train_pos = _build_pos_dict(train_df)
    cold_pos  = _build_pos_dict(cold_df)

    users_all = list(cold_pos.keys())
    users = users_all if users_subset is None else [u for u in users_all if u in users_subset]
    if len(users) == 0:
        return 0.0, 0.0

    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    recall_sum, ndcg_sum, n_users = 0.0, 0.0, 0
    for u in users:
        gt = cold_pos.get(u, set())
        if not gt:
            continue

        # 후보군 구성: train에서 본 것 제거
        u_train = train_pos.get(u, set())
        if u_train:
            mask = torch.ones(n_items, dtype=torch.bool, device=device)
            idx = torch.tensor(list(u_train), dtype=torch.long, device=device)
            mask[idx] = False
            candidates = all_items[mask]
        else:
            candidates = all_items

        # ★ 정답(gt)을 후보군에 반드시 포함
        gt_tensor = torch.tensor(sorted(gt), dtype=torch.long, device=device)
        candidates = torch.unique(torch.cat([candidates, gt_tensor]))

        # 점수 계산(배치)
        scores_list = []
        num_c = candidates.numel()
        for s in range(0, num_c, batch_items):
            e = min(s + batch_items, num_c)
            items_chunk = candidates[s:e]
            users_chunk = torch.full((items_chunk.numel(),), u, dtype=torch.long, device=device)
            sc = scorer_fn(users_chunk, items_chunk).reshape(-1).to(torch.float32)
            scores_list.append(sc)

        if not scores_list:
            continue
        scores = torch.cat(scores_list, dim=0)

        K_eff = min(K, scores.numel())
        if K_eff <= 0:
            continue

        _, topk_rel_idx = torch.topk(scores, k=K_eff, largest=True, sorted=True)
        topk_items = candidates[topk_rel_idx].contiguous().to("cpu").tolist()

        # Recall@K
        hit_cnt = sum((it in gt) for it in topk_items)
        recall_u = hit_cnt / max(1, len(gt))

        # NDCG@K
        dcg = 0.0
        for rank, it in enumerate(topk_items):
            if it in gt:
                dcg += 1.0 / math.log2(rank + 2.0)
        idcg = sum(1.0 / math.log2(r + 2.0) for r in range(min(K_eff, len(gt))))
        ndcg_u = (dcg / idcg) if idcg > 0 else 0.0

        recall_sum += recall_u
        ndcg_sum += ndcg_u
        n_users += 1

    if n_users == 0:
        return 0.0, 0.0
    return float(recall_sum / n_users), float(ndcg_sum / n_users)