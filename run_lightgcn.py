# ---------------------------------------------
# run_lightgcn.py  (UPDATED: CF Teacher + Content->Embedding Student + TS Eval)
# - mode=cf:        기존 LightGCN 학습 + (옵션) item-item + (옵션) align, 그리고 teacher embedding 저장
# - mode=student:   teacher item embedding을 target으로 콘텐츠(genre)->embedding 회귀(MLP) 학습
# - mode=ts_eval:   teacher의 user embedding + (warm item=teacher, cold item=student 생성)로 cold ranking 평가
#
# NOTE:
# - items.csv에는 genre_* one-hot이 있어야 함.
# - split_dir: train/valid/test_cold.csv 가 있어야 함.
# ---------------------------------------------

import argparse, os, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.metrics import (
    eval_rmse_mae,
    eval_warm_ndcg_at_k,
    eval_cold_recall_ndcg_at_k,
)

# -----------------------------
# 데이터셋 (u,i,r) triplet
# -----------------------------
class RatingsDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)
    def __len__(self): return self.r.shape[0]
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

# -----------------------------
# 속성 로딩 (users.csv / items.csv)
# -----------------------------
def build_user_item_attr_mats(processed_dir: Path):
    """
    users.csv / items.csv -> dense attribute matrices
    - users: [age(min-max), gender one-hot(F,M), occupation one-hot]
    - items: genre_* one-hot
    반환: (user_attr_mat [U,Du], item_attr_mat [I,Di]) (np.float32)
    """
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")

    # Users
    user_ids = users["user_id"].astype(int).tolist()

    age = users["age"].astype(float).values
    age = ((age - np.nanmin(age)) / (np.nanmax(age) - np.nanmin(age) + 1e-8)).reshape(-1,1).astype(np.float32)

    genders = users["gender"].fillna("U").astype(str).values
    g_vocab = ["F","M"]
    g_idx = {g:i for i,g in enumerate(g_vocab)}
    g_oh = np.zeros((len(users), len(g_vocab)), dtype=np.float32)
    for r,g in enumerate(genders):
        if g in g_idx: g_oh[r, g_idx[g]] = 1.0

    occ_vals = users["occupation"].fillna("other").astype(str).values
    occ_vocab = sorted(pd.Series(occ_vals).unique().tolist())
    occ_idx = {o:i for i,o in enumerate(occ_vocab)}
    occ_oh = np.zeros((len(users), len(occ_vocab)), dtype=np.float32)
    for r,o in enumerate(occ_vals):
        occ_oh[r, occ_idx[o]] = 1.0

    user_attr = np.concatenate([age, g_oh, occ_oh], axis=1).astype(np.float32)

    max_uid = max(user_ids)
    U = max_uid + 1
    user_attr_mat = np.zeros((U, user_attr.shape[1]), dtype=np.float32)
    for r in range(len(users)):
        u_orig = int(users.iloc[r]["user_id"])
        user_attr_mat[u_orig] = user_attr[r]

    # Items
    item_ids = items["item_id"].astype(int).tolist()
    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    if not genre_cols:
        raise RuntimeError("items.csv에서 genre_* 컬럼을 찾지 못했습니다.")
    item_attr = items[genre_cols].astype(np.float32).values

    max_iid = max(item_ids)
    I = max_iid + 1
    item_attr_mat = np.zeros((I, item_attr.shape[1]), dtype=np.float32)
    for r in range(len(items)):
        it_orig = int(items.iloc[r]["item_id"])
        item_attr_mat[it_orig] = item_attr[r]

    return user_attr_mat, item_attr_mat

# -----------------------------
# item-item kNN 로더
# -----------------------------
def load_item_item_knn_csv(path: str, n_items: int):
    """
    item_item_knn.csv columns: item_i, item_j, sim
    여기서 item_i/item_j 는 "모델 내부 인덱스(0-based)" 기준이어야 함.
    """
    df = pd.read_csv(path)
    required = {"item_i","item_j"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{path} 에 item_i,item_j 컬럼이 필요합니다. 현재 컬럼: {list(df.columns)}")

    df = df[(df["item_i"] >= 0) & (df["item_i"] < n_items) &
            (df["item_j"] >= 0) & (df["item_j"] < n_items)]

    edges = df[["item_i","item_j"]].to_numpy(dtype=np.int64)
    sims = df["sim"].to_numpy(dtype=np.float32) if "sim" in df.columns else None
    return edges, sims

def build_item_item_neighbor_map(
    ii_edges: np.ndarray,
    ii_sims: Optional[np.ndarray],
    warm_items_set: set,
    n_items: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    item -> [(neighbor, sim), ...]
    - alignment target는 'warm neighbor'만 사용(학습된 anchor 역할)
    - ii_sims 없으면 sim=1.0
    """
    nbrs: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_items)}
    if ii_edges is None or len(ii_edges) == 0:
        return nbrs

    if ii_sims is None:
        ii_sims = np.ones((len(ii_edges),), dtype=np.float32)

    for (a, b), s in zip(ii_edges, ii_sims):
        a = int(a); b = int(b)
        w = float(s)

        if b in warm_items_set:
            nbrs[a].append((b, w))
        if a in warm_items_set:
            nbrs[b].append((a, w))

    return nbrs

# -----------------------------
# LightGCN 모델
# -----------------------------
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, layers, A_norm_sparse,
                 use_attrs=False,
                 user_attr_mat=None, item_attr_mat=None, attr_dim_user=0, attr_dim_item=0,
                 gate_init=0.3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.layers = layers
        self.A = A_norm_sparse  # torch.sparse_coo_tensor (shape: [U+I, U+I])

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.use_attrs = use_attrs
        if use_attrs:
            self.register_buffer("user_attr_buf", torch.from_numpy(user_attr_mat).float())
            self.register_buffer("item_attr_buf", torch.from_numpy(item_attr_mat).float())
            self.user_attr_fc = nn.Linear(attr_dim_user, dim)
            self.item_attr_fc = nn.Linear(attr_dim_item, dim)
            nn.init.xavier_uniform_(self.user_attr_fc.weight); nn.init.zeros_(self.user_attr_fc.bias)
            nn.init.xavier_uniform_(self.item_attr_fc.weight); nn.init.zeros_(self.item_attr_fc.bias)
            self.gate_u = nn.Parameter(torch.tensor(float(gate_init)))
            self.gate_i = nn.Parameter(torch.tensor(float(gate_init)))

    def _initial_emb(self):
        Ue = self.user_emb.weight
        Ie = self.item_emb.weight
        if self.use_attrs:
            Ua = self.user_attr_fc(self.user_attr_buf)
            Ia = self.item_attr_fc(self.item_attr_buf)
            Ue = Ue + torch.sigmoid(self.gate_u) * Ua
            Ie = Ie + torch.sigmoid(self.gate_i) * Ia

        x0 = torch.zeros(self.n_users + self.n_items, self.dim, device=Ue.device, dtype=Ue.dtype)
        x0[:self.n_users] = Ue
        x0[self.n_users:] = Ie
        return x0

    def compute_embeddings(self):
        xk = self._initial_emb()
        h = xk.clone()
        for _ in range(self.layers):
            xk = torch.sparse.mm(self.A, xk)
            h = h + xk
        h = h / float(self.layers + 1)
        h_user = h[:self.n_users]
        h_item = h[self.n_users:]
        return h_user, h_item

    @staticmethod
    def score_from_embeddings(h_user: torch.Tensor, h_item: torch.Tensor,
                             users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_e = h_user[users]
        i_e = h_item[items]
        return (u_e * i_e).sum(dim=-1)

    def forward(self, users, items):
        h_user, h_item = self.compute_embeddings()
        return self.score_from_embeddings(h_user, h_item, users, items)

    def predict(self, users, items):
        return self.forward(users, items)

# -----------------------------
# 인접행렬(정규화) 만들기: user-item + (optional) item-item
# -----------------------------
def build_lightgcn_norm_adj(n_users, n_items, train_df, device,
                           use_item_item=False,
                           item_item_edges=None,
                           item_item_sims=None,
                           ii_strength=1.0):
    U = int(n_users); I = int(n_items)
    N = U + I

    u = train_df["user_id"].to_numpy(dtype=np.int64)
    i = train_df["item_id"].to_numpy(dtype=np.int64)

    ui_src = u
    ui_dst = i + U
    iu_src = i + U
    iu_dst = u

    rows_list = [ui_src, iu_src]
    cols_list = [ui_dst, iu_dst]
    vals_list = [np.ones_like(ui_src, dtype=np.float32),
                 np.ones_like(iu_src, dtype=np.float32)]

    if use_item_item and (item_item_edges is not None) and (len(item_item_edges) > 0):
        a = item_item_edges[:, 0].astype(np.int64)
        b = item_item_edges[:, 1].astype(np.int64)

        ab_src = a + U; ab_dst = b + U
        ba_src = b + U; ba_dst = a + U

        rows_list += [ab_src, ba_src]
        cols_list += [ab_dst, ba_dst]

        if item_item_sims is None:
            w = np.ones_like(a, dtype=np.float32) * float(ii_strength)
        else:
            w = item_item_sims.astype(np.float32) * float(ii_strength)

        vals_list += [w, w]

    rows = np.concatenate(rows_list, axis=0).astype(np.int64)
    cols = np.concatenate(cols_list, axis=0).astype(np.int64)
    vals = np.concatenate(vals_list, axis=0).astype(np.float32)

    idx_np = np.stack([rows, cols], axis=0)
    indices = torch.from_numpy(idx_np).to(device)
    values = torch.from_numpy(vals).to(device)

    A = torch.sparse_coo_tensor(indices, values, torch.Size([N, N]), device=device).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    r, c = A.indices()
    norm_vals = A.values() * (deg_inv_sqrt[r] * deg_inv_sqrt[c])
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_vals, A.shape, device=device).coalesce()
    return A_norm

# -----------------------------
# Cold item alignment loss
# -----------------------------
def cold_alignment_loss(
    h_item: torch.Tensor,
    cold_items: np.ndarray,
    neighbor_map: Dict[int, List[Tuple[int, float]]],
    align_k: int,
    align_batch_items: int,
    use_sims: bool,
    device: str,
) -> torch.Tensor:
    if cold_items is None or len(cold_items) == 0:
        return torch.tensor(0.0, device=device)

    if align_batch_items <= 0 or align_batch_items >= len(cold_items):
        sampled = cold_items
    else:
        idx = np.random.choice(len(cold_items), size=align_batch_items, replace=False)
        sampled = cold_items[idx]

    losses = []
    for it in sampled:
        it = int(it)
        nbrs = neighbor_map.get(it, [])
        if not nbrs:
            continue

        if use_sims:
            nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)
        else:
            nbrs_sorted = nbrs
        nbrs_k = nbrs_sorted[:max(1, align_k)]

        nbr_ids = torch.tensor([n for (n, _) in nbrs_k], dtype=torch.long, device=device)
        nbr_emb = h_item[nbr_ids]

        if use_sims:
            w = torch.tensor([s for (_, s) in nbrs_k], dtype=torch.float32, device=device)
            w = torch.clamp(w, min=0.0)
            w = w / (w.sum() + 1e-12)
            target = (w.unsqueeze(1) * nbr_emb).sum(dim=0)
        else:
            target = nbr_emb.mean(dim=0)

        target = target.detach()
        e_i = h_item[it]
        losses.append(F.mse_loss(e_i, target))

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()

# -----------------------------
# Student: 콘텐츠(genre)->임베딩 MLP
# -----------------------------
class ContentMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------
# 고정 임베딩으로 RMSE/MAE 계산
# -----------------------------
@torch.no_grad()
def eval_rmse_mae_fixed(h_user: torch.Tensor, h_item: torch.Tensor, loader: DataLoader, device: str) -> Tuple[float,float]:
    preds, trues = [], []
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        p = (h_user[u] * h_item[i]).sum(dim=-1)
        preds.append(p.detach().cpu().reshape(-1).numpy())
        trues.append(r.detach().cpu().reshape(-1).numpy())
    preds = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    trues = np.concatenate(trues) if trues else np.array([], dtype=np.float32)
    if preds.size == 0:
        return 0.0, 0.0
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    mae  = float(np.abs(preds - trues).mean())
    return rmse, mae

def main():
    ap = argparse.ArgumentParser()

    # ✅ mode 추가
    ap.add_argument("--mode", choices=["cf","student","ts_eval"], default="cf")

    ap.add_argument("--split", default="splits/seed42_u0.1_i0.0")
    ap.add_argument("--attrs", choices=["OFF","ON"], default="OFF")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate_init", type=float, default=0.3)
    ap.add_argument("--eval_k", type=int, default=100)
    ap.add_argument("--out", default="results/metrics_light.csv")

    # item-item
    ap.add_argument("--use_item_item", choices=["OFF","ON"], default="OFF")
    ap.add_argument("--item_item_path", default="data/processed/item_item_knn.csv")
    ap.add_argument("--ii_strength", type=float, default=1.0)

    # align
    ap.add_argument("--use_align", choices=["OFF","ON"], default="OFF")
    ap.add_argument("--align_lambda", type=float, default=0.1)
    ap.add_argument("--align_k", type=int, default=20)
    ap.add_argument("--align_batch_items", type=int, default=512)
    ap.add_argument("--align_use_sims", choices=["OFF","ON"], default="ON")

    # ✅ teacher 저장/로드 경로
    ap.add_argument("--teacher_path", default="results/teacher_emb.pt")

    # ✅ student 설정
    ap.add_argument("--student_path", default="results/student_mlp.pt")
    ap.add_argument("--student_hid", type=int, default=256)
    ap.add_argument("--student_dropout", type=float, default=0.0)
    ap.add_argument("--student_epochs", type=int, default=50)
    ap.add_argument("--student_lr", type=float, default=1e-3)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_attrs = (args.attrs.upper() == "ON")
    use_item_item = (args.use_item_item.upper() == "ON")
    use_align = (args.use_align.upper() == "ON")
    align_use_sims = (args.align_use_sims.upper() == "ON")

    split_dir = Path(args.split)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = int(max(tr.user_id.max(), va.user_id.max(), te.user_id.max()) + 1)
    n_items = int(max(tr.item_id.max(), va.item_id.max(), te.item_id.max()) + 1)

    # attrs mats (필요 시)
    user_attr_mat = None; item_attr_mat = None
    Du = 0; Di = 0
    if use_attrs or args.mode in ["student","ts_eval"]:
        # student/ts_eval은 item genre 필요
        user_attr_mat, item_attr_mat = build_user_item_attr_mats(Path("data/processed"))
        user_attr_mat = user_attr_mat[:n_users]
        item_attr_mat = item_attr_mat[:n_items]
        Du = user_attr_mat.shape[1]
        Di = item_attr_mat.shape[1]

    # item-item edges (필요 시)
    ii_edges, ii_sims = (None, None)
    if use_item_item:
        ii_edges, ii_sims = load_item_item_knn_csv(args.item_item_path, n_items)

    # -----------------------------------------
    # MODE 1) CF Teacher 학습 + 임베딩 저장
    # -----------------------------------------
    if args.mode == "cf":
        A_norm = build_lightgcn_norm_adj(
            n_users, n_items, tr, device,
            use_item_item=use_item_item,
            item_item_edges=ii_edges,
            item_item_sims=ii_sims,
            ii_strength=args.ii_strength
        )

        model = LightGCN(
            n_users=n_users, n_items=n_items, dim=args.dim, layers=args.layers,
            A_norm_sparse=A_norm,
            use_attrs=use_attrs,
            user_attr_mat=user_attr_mat if use_attrs else None,
            item_attr_mat=item_attr_mat if use_attrs else None,
            attr_dim_user=Du, attr_dim_item=Di,
            gate_init=args.gate_init,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        tr_loader = DataLoader(RatingsDS(tr), batch_size=args.bs, shuffle=True, num_workers=0)
        va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)
        te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)

        cold_items = None
        neighbor_map = None
        if use_align:
            if not use_item_item:
                raise RuntimeError("--use_align ON을 쓰려면 --use_item_item ON이 필요합니다.")

            tr_items_set = set(tr["item_id"].unique().tolist())
            te_items_set = set(te["item_id"].unique().tolist())
            cold_only = sorted(list(te_items_set - tr_items_set))
            cold_items = np.array(cold_only, dtype=np.int64)

            neighbor_map = build_item_item_neighbor_map(
                ii_edges, ii_sims, warm_items_set=tr_items_set, n_items=n_items
            )
            print(f"[ALIGN] cold_items={len(cold_items)} | align_lambda={args.align_lambda} | "
                  f"align_k={args.align_k} | align_batch_items={args.align_batch_items} | use_sims={align_use_sims}")

        best = {"rmse": 1e9, "mae": 1e9, "ndcg": -1.0, "ep": 0, "state": None}

        for ep in range(1, args.epochs + 1):
            model.train()
            for u, i, r in tr_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)

                h_user, h_item = model.compute_embeddings()
                pred = model.score_from_embeddings(h_user, h_item, u, i)

                loss = loss_fn(pred, r)

                if use_align and (args.align_lambda > 0):
                    align_loss = cold_alignment_loss(
                        h_item=h_item,
                        cold_items=cold_items,
                        neighbor_map=neighbor_map,
                        align_k=args.align_k,
                        align_batch_items=args.align_batch_items,
                        use_sims=align_use_sims,
                        device=device,
                    )
                    loss = loss + float(args.align_lambda) * align_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            vrmse, vmae = eval_rmse_mae(model, va_loader, device)
            vndcg = eval_warm_ndcg_at_k(model, tr, va, n_items, K=10, device=device)
            print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

            if vrmse < best["rmse"]:
                best.update({"rmse": vrmse, "mae": vmae, "ndcg": vndcg, "ep": ep})
                best["state"] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k,v in model.state_dict().items()}

        # best state 로드 후 embedding 저장
        if best["state"] is not None:
            model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k,v in best["state"].items()}, strict=True)

        with torch.no_grad():
            model.eval()
            h_user, h_item = model.compute_embeddings()
            payload = {
                "n_users": n_users,
                "n_items": n_items,
                "dim": args.dim,
                "layers": args.layers,
                "split": str(split_dir),
                "h_user": h_user.detach().cpu(),
                "h_item": h_item.detach().cpu(),
                "train_items": sorted(list(set(tr["item_id"].unique().tolist()))),
            }
            os.makedirs(Path(args.teacher_path).parent, exist_ok=True)
            torch.save(payload, args.teacher_path)
            print(f"[SAVE] teacher embeddings -> {args.teacher_path}")

        # cold eval (기존 방식)
        cold_rmse, cold_mae = eval_rmse_mae(model, te_loader, device)
        cold_recallK, cold_ndcgK = eval_cold_recall_ndcg_at_k(model, tr, te, n_items, K=args.eval_k, device=device)
        print(f"[COLD TEST] RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | Recall@{args.eval_k}={cold_recallK:.6f} NDCG@{args.eval_k}={cold_ndcgK:.6f} (best Ep {best['ep']})")

        # csv 저장(기존 컬럼 유지)
        os.makedirs("results", exist_ok=True)
        need_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow([
                    "mode","model","attrs","use_item_item","ii_strength","split_dir","dim","layers","epochs",
                    "use_align","align_lambda","align_k","align_batch_items","align_use_sims",
                    "valid_rmse","valid_mae","valid_ndcg10",
                    "cold_rmse","cold_mae","cold_recallK","cold_ndcgK","eval_k",
                    "teacher_path"
                ])
            w.writerow([
                "cf","LightGCN", args.attrs, args.use_item_item, args.ii_strength, str(split_dir), args.dim, args.layers, best["ep"],
                args.use_align, args.align_lambda, args.align_k, args.align_batch_items, args.align_use_sims,
                best["rmse"], best["mae"], best["ndcg"],
                cold_rmse, cold_mae, cold_recallK, cold_ndcgK, args.eval_k,
                args.teacher_path
            ])
        return

    # -----------------------------------------
    # MODE 2) Student 학습: 콘텐츠->Teacher 임베딩 회귀
    # -----------------------------------------
    if args.mode == "student":
        if not os.path.exists(args.teacher_path):
            raise RuntimeError(f"teacher_path가 없습니다: {args.teacher_path} (먼저 --mode cf로 teacher를 저장하세요.)")

        teacher = torch.load(args.teacher_path, map_location="cpu")
        h_item_T = teacher["h_item"]  # [I,dim]
        train_items = set(teacher.get("train_items", []))

        # warm item만 학습 (teacher target 있는 것들)
        warm_items = sorted([i for i in range(n_items) if i in train_items])
        if len(warm_items) == 0:
            raise RuntimeError("warm_items가 0입니다. teacher의 train_items 저장을 확인하세요.")

        X = torch.from_numpy(item_attr_mat).float()          # [I,Di]
        Y = h_item_T.float()                                 # [I,dim]

        Xw = X[warm_items]
        Yw = Y[warm_items]

        ds = torch.utils.data.TensorDataset(Xw, Yw)
        dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

        modelS = ContentMLP(in_dim=Di, out_dim=args.dim, hid=args.student_hid, dropout=args.student_dropout).to(device)
        optS = torch.optim.Adam(modelS.parameters(), lr=args.student_lr, weight_decay=1e-6)

        best_loss = 1e18
        best_state = None

        for ep in range(1, args.student_epochs + 1):
            modelS.train()
            losses = []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = modelS(xb)
                loss = F.mse_loss(pred, yb)
                optS.zero_grad()
                loss.backward()
                optS.step()
                losses.append(loss.item())
            avg = float(np.mean(losses)) if losses else 0.0
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.detach().cpu() for k,v in modelS.state_dict().items()}
            if ep % 5 == 0 or ep == 1:
                print(f"[Student Ep {ep:03d}] emb_mse={avg:.6f}")

        if best_state is not None:
            modelS.load_state_dict({k: v.to(device) for k,v in best_state.items()}, strict=True)

        os.makedirs(Path(args.student_path).parent, exist_ok=True)
        torch.save({
            "in_dim": Di,
            "out_dim": args.dim,
            "hid": args.student_hid,
            "dropout": args.student_dropout,
            "state": {k: v.detach().cpu() for k,v in modelS.state_dict().items()},
            "teacher_path": args.teacher_path,
            "split": str(split_dir),
            "best_emb_mse": best_loss,
        }, args.student_path)
        print(f"[SAVE] student MLP -> {args.student_path} (best_emb_mse={best_loss:.6f})")
        return

    # -----------------------------------------
    # MODE 3) TS 평가: warm item=Teacher, cold item=Student 생성
    # -----------------------------------------
    if args.mode == "ts_eval":
        if not os.path.exists(args.teacher_path):
            raise RuntimeError(f"teacher_path가 없습니다: {args.teacher_path}")
        if not os.path.exists(args.student_path):
            raise RuntimeError(f"student_path가 없습니다: {args.student_path}")

        teacher = torch.load(args.teacher_path, map_location="cpu")
        h_user_T = teacher["h_user"].float()  # [U,dim]
        h_item_T = teacher["h_item"].float()  # [I,dim]
        train_items = set(teacher.get("train_items", []))

        student_ckpt = torch.load(args.student_path, map_location="cpu")
        modelS = ContentMLP(
            in_dim=student_ckpt["in_dim"],
            out_dim=student_ckpt["out_dim"],
            hid=student_ckpt["hid"],
            dropout=student_ckpt["dropout"],
        ).to(device)
        modelS.load_state_dict({k: v.to(device) for k,v in student_ckpt["state"].items()}, strict=True)
        modelS.eval()

        # cold items = test_cold - train
        tr_items_set = set(tr["item_id"].unique().tolist())
        te_items_set = set(te["item_id"].unique().tolist())
        cold_only = sorted(list(te_items_set - tr_items_set))
        cold_items = set(cold_only)

        # item embedding 생성: warm=teacher, cold=student(content)
        X = torch.from_numpy(item_attr_mat).float().to(device)  # [I,Di]
        with torch.no_grad():
            cold_emb = modelS(X)  # [I,dim] (모든 아이템 생성해두고 cold만 사용)
        h_item_mix = h_item_T.clone()  # cpu 텐서
        # cpu 텐서를 한 번에 만들기 위해 cold_emb를 cpu로
        cold_emb_cpu = cold_emb.detach().cpu()

        for i in cold_only:
            if 0 <= i < n_items:
                h_item_mix[i] = cold_emb_cpu[i]

        # device로 올려서 scorer 구성
        h_user = h_user_T.to(device)
        h_item = h_item_mix.to(device)

        # scorer: (users, items) -> scores
        def scorer(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
            return (h_user[users] * h_item[items]).sum(dim=-1)

        # loaders for rmse/mae
        va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)
        te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)

        # warm RMSE/MAE는 fixed embedding으로 계산(참고)
        vrmse, vmae = eval_rmse_mae_fixed(h_user, h_item, va_loader, device=device)

        # warm ranking (참고)
        warm_ndcg10 = eval_warm_ndcg_at_k(
            scorer, tr, va, n_items, K=10, device=device, scorer=scorer
        )

        cold_rmse, cold_mae = eval_rmse_mae_fixed(h_user, h_item, te_loader, device=device)
        cold_recallK, cold_ndcgK = eval_cold_recall_ndcg_at_k(
            scorer, tr, te, n_items, K=args.eval_k, device=device, scorer=scorer
        )

        print(f"[TS EVAL] valid RMSE={vrmse:.4f} MAE={vmae:.4f} warmNDCG@10={warm_ndcg10:.4f}")
        print(f"[TS EVAL] cold  RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | Recall@{args.eval_k}={cold_recallK:.6f} NDCG@{args.eval_k}={cold_ndcgK:.6f}")

        os.makedirs("results", exist_ok=True)
        need_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow([
                    "mode","model","split_dir","dim",
                    "teacher_path","student_path",
                    "valid_rmse","valid_mae","warm_ndcg10",
                    "cold_rmse","cold_mae","cold_recallK","cold_ndcgK","eval_k",
                    "n_cold_items"
                ])
            w.writerow([
                "ts_eval","LightGCN+Student", str(split_dir), args.dim,
                args.teacher_path, args.student_path,
                vrmse, vmae, warm_ndcg10,
                cold_rmse, cold_mae, cold_recallK, cold_ndcgK, args.eval_k,
                len(cold_only)
            ])
        return

if __name__ == "__main__":
    main()