# run_ts_itemcold.py
# ------------------------------------------------------------
# Teacher–Student for item-cold:
#  - Teacher: (LightGCN/MF/GCN) 학습 -> (h_user, h_item) 저장
#  - Student: item content(genre one-hot) -> teacher h_item 회귀
#  - Eval: cold item embedding을 student 예측으로 교체 후 Recall/NDCG 측정
#
# + Uncertainty-aware weighting (MC Dropout)
#   - Teacher에서 MC Dropout으로 item embedding sigma 추정
#   - Student distillation loss에 exp(-alpha*sigma) weight 적용
# ------------------------------------------------------------

import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.metrics import (
    eval_rmse_mae,
    eval_warm_ndcg_at_k,
    eval_cold_recall_ndcg_at_k,
)

# -----------------------------
# Dataset
# -----------------------------
class RatingsDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return self.r.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

# -----------------------------
# Load item genres (content)
# items.csv: item_id(1-based), genre_* one-hot
# returns item_feat_mat [n_items+1, D] with row 0 = zeros
# -----------------------------
def load_item_genre_features(items_csv: str, n_items: int):
    items = pd.read_csv(items_csv)
    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    assert len(genre_cols) > 0, "items.csv에 genre_* 컬럼이 없음"

    D = len(genre_cols)
    feat = np.zeros((n_items + 1, D), dtype=np.float32)

    for _, row in items.iterrows():
        iid = int(row["item_id"])
        if 0 <= iid <= n_items:
            feat[iid] = row[genre_cols].astype(np.float32).values

    return feat, D

# -----------------------------
# item-item kNN (optional for teacher)
# expects 1-based item_i/item_j already
# -----------------------------
def load_item_item_knn_csv(path: str, n_items: int):
    df = pd.read_csv(path)
    req = {"item_i", "item_j"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"{path} needs item_i,item_j. got={list(df.columns)}")

    df = df[(df["item_i"] >= 0) & (df["item_i"] <= n_items) &
            (df["item_j"] >= 0) & (df["item_j"] <= n_items)]
    edges = df[["item_i", "item_j"]].to_numpy(dtype=np.int64)
    sims = df["sim"].to_numpy(dtype=np.float32) if "sim" in df.columns else None
    return edges, sims

# -----------------------------
# build normalized adjacency (user-item + optional item-item)
# IDs assumed 1-based, but we allocate [0..n_users] and [0..n_items]
# row 0 is unused but harmless
# -----------------------------
def build_lightgcn_norm_adj(
    n_users, n_items, train_df, device,
    use_item_item=False,
    item_item_edges=None,
    item_item_sims=None,
    ii_strength=1.0
):
    U = int(n_users) + 1
    I = int(n_items) + 1
    N = U + I

    u = train_df["user_id"].to_numpy(dtype=np.int64)
    i = train_df["item_id"].to_numpy(dtype=np.int64)

    ui_src = u
    ui_dst = i + U
    iu_src = i + U
    iu_dst = u

    rows_list = [ui_src, iu_src]
    cols_list = [ui_dst, iu_dst]
    vals_list = [
        np.ones_like(ui_src, dtype=np.float32),
        np.ones_like(iu_src, dtype=np.float32),
    ]

    if use_item_item and item_item_edges is not None and len(item_item_edges) > 0:
        a = item_item_edges[:, 0].astype(np.int64)
        b = item_item_edges[:, 1].astype(np.int64)

        ab_src = a + U
        ab_dst = b + U
        ba_src = b + U
        ba_dst = a + U

        rows_list += [ab_src, ba_src]
        cols_list += [ab_dst, ba_dst]

        if item_item_sims is None:
            w = np.ones_like(a, dtype=np.float32) * float(ii_strength)
        else:
            w = item_item_sims.astype(np.float32) * float(ii_strength)

        vals_list += [w, w]

    rows = np.concatenate(rows_list).astype(np.int64)
    cols = np.concatenate(cols_list).astype(np.int64)
    vals = np.concatenate(vals_list).astype(np.float32)

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
    return A_norm, U, I

# ============================================================
# Teacher models
# ============================================================

# -----------------------------
# LightGCN (teacher)
# CHANGED: MC Dropout 지원을 위해 embedding dropout 추가
# -----------------------------
class LightGCN(nn.Module):
    def __init__(self, U_size, I_size, dim, layers, A_norm_sparse, emb_dropout=0.0):
        super().__init__()
        self.U = U_size
        self.I = I_size
        self.dim = dim
        self.layers = layers
        self.A = A_norm_sparse

        self.user_emb = nn.Embedding(self.U, dim)
        self.item_emb = nn.Embedding(self.I, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.emb_dropout = float(emb_dropout)
        self.drop = nn.Dropout(self.emb_dropout)

    def compute_embeddings(self, mc_dropout: bool = False):
        x0 = torch.zeros(self.U + self.I, self.dim, device=self.user_emb.weight.device)
        x0[:self.U] = self.user_emb.weight
        x0[self.U:] = self.item_emb.weight

        # ADDED: MC Dropout용. (train 모드에서만 dropout 작동)
        if mc_dropout and self.emb_dropout > 0:
            x0 = self.drop(x0)

        xk = x0
        h = x0.clone()
        for _ in range(self.layers):
            xk = torch.sparse.mm(self.A, xk)
            if mc_dropout and self.emb_dropout > 0:
                xk = self.drop(xk)
            h = h + xk
        h = h / float(self.layers + 1)

        h_user = h[:self.U]
        h_item = h[self.U:]
        return h_user, h_item

    @staticmethod
    def score(h_user, h_item, users, items):
        return (h_user[users] * h_item[items]).sum(dim=-1)

    def forward(self, users, items):
        h_user, h_item = self.compute_embeddings(mc_dropout=False)
        return self.score(h_user, h_item, users, items)

# -----------------------------
# MF (teacher)
# CHANGED: MC Dropout용 embedding dropout 추가
# -----------------------------
class MFTeacher(nn.Module):
    def __init__(self, U_size, I_size, dim, emb_dropout=0.0):
        super().__init__()
        self.U = U_size
        self.I = I_size
        self.dim = dim

        self.user_emb = nn.Embedding(self.U, dim)
        self.item_emb = nn.Embedding(self.I, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.emb_dropout = float(emb_dropout)
        self.drop = nn.Dropout(self.emb_dropout)

    def compute_embeddings(self, mc_dropout: bool = False):
        hu = self.user_emb.weight
        hi = self.item_emb.weight
        if mc_dropout and self.emb_dropout > 0:
            hu = self.drop(hu)
            hi = self.drop(hi)
        return hu, hi

    @staticmethod
    def score(h_user, h_item, users, items):
        return (h_user[users] * h_item[items]).sum(dim=-1)

    def forward(self, users, items):
        h_user, h_item = self.compute_embeddings(mc_dropout=False)
        return self.score(h_user, h_item, users, items)

# -----------------------------
# GCN (teacher) - bipartite graph + Linear/ReLU/Dropout
# CHANGED: compute_embeddings에 mc_dropout 플래그 추가(동일 인터페이스)
# -----------------------------
class GCNTeacher(nn.Module):
    def __init__(self, U_size, I_size, dim, layers, A_norm_sparse, dropout=0.0):
        super().__init__()
        self.U = U_size
        self.I = I_size
        self.dim = dim
        self.layers = layers
        self.A = A_norm_sparse
        self.dropout = float(dropout)

        self.user_emb = nn.Embedding(self.U, dim)
        self.item_emb = nn.Embedding(self.I, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.W = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        for w in self.W:
            nn.init.xavier_uniform_(w.weight)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

    def compute_embeddings(self, mc_dropout: bool = False):
        x = torch.zeros(self.U + self.I, self.dim, device=self.user_emb.weight.device)
        x[:self.U] = self.user_emb.weight
        x[self.U:] = self.item_emb.weight

        for li in range(self.layers):
            x = torch.sparse.mm(self.A, x)
            x = self.W[li](x)
            x = self.act(x)
            # dropout은 mc_dropout일 때만 “의도적으로” 켬
            if mc_dropout and self.dropout > 0:
                x = self.drop(x)

        h_user = x[:self.U]
        h_item = x[self.U:]
        return h_user, h_item

    @staticmethod
    def score(h_user, h_item, users, items):
        return (h_user[users] * h_item[items]).sum(dim=-1)

    def forward(self, users, items):
        h_user, h_item = self.compute_embeddings(mc_dropout=False)
        return self.score(h_user, h_item, users, items)

# -----------------------------
# Student: MLP(content -> item embedding)
# -----------------------------
class ContentMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# scorer helper
# -----------------------------
def make_scorer(h_user, h_item_final):
    def scorer(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (h_user[users] * h_item_final[items]).sum(dim=-1)
    return scorer

# -----------------------------
# Sampled evaluation helpers
# -----------------------------
def _build_user_hist(df: pd.DataFrame):
    hist = {}
    for u, g in df.groupby("user_id"):
        hist[int(u)] = set(g["item_id"].astype(int).tolist())
    return hist

@torch.no_grad()
def eval_sampled_recall_ndcg_at_k(
    scorer,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    n_items_plus1: int,
    K: int,
    n_neg: int = 99,
    seed: int = 2025,
    device: str = "cpu",
):
    rng = np.random.default_rng(seed)
    train_hist = _build_user_hist(train_df)

    recalls = []
    ndcgs = []

    users_np = eval_df["user_id"].to_numpy(dtype=np.int64)
    pos_items_np = eval_df["item_id"].to_numpy(dtype=np.int64)

    for u, pos in zip(users_np, pos_items_np):
        u = int(u)
        pos = int(pos)

        banned = set(train_hist.get(u, set()))
        banned.add(pos)

        negs = []
        while len(negs) < n_neg:
            cand = int(rng.integers(1, n_items_plus1))
            if cand not in banned:
                negs.append(cand)

        cand_items = [pos] + negs
        items_t = torch.tensor(cand_items, dtype=torch.long, device=device)
        users_t = torch.full((len(cand_items),), u, dtype=torch.long, device=device)

        scores = scorer(users_t, items_t)
        order = torch.argsort(scores, descending=True)

        rank = (order == 0).nonzero(as_tuple=False).item()  # 0-based

        recalls.append(1.0 if rank < K else 0.0)
        ndcgs.append(1.0 / np.log2(rank + 2.0) if rank < K else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))

# ============================================================
# Uncertainty (MC Dropout) utilities
# ============================================================
@torch.no_grad()
def estimate_item_uncertainty_mc_dropout(teacher, T: int, device: str):
    """
    teacher.compute_embeddings(mc_dropout=True) 를 T번 호출해서 item embedding sigma 추정.
    sigma는 per-item scalar로 반환 (embedding norm의 std).
    return:
      h_user_mu: [U, dim]
      h_item_mu: [I, dim]
      sigma_item: [I]  (float32)
    """
    # MC Dropout을 켜려면 teacher를 train 모드로 두되, gradient는 no_grad로 막는다.
    teacher.train()

    h_user_list = []
    h_item_list = []
    for _ in range(T):
        hu, hi = teacher.compute_embeddings(mc_dropout=True)
        h_user_list.append(hu.unsqueeze(0))
        h_item_list.append(hi.unsqueeze(0))

    H_u = torch.cat(h_user_list, dim=0)  # [T, U, dim]
    H_i = torch.cat(h_item_list, dim=0)  # [T, I, dim]

    h_user_mu = H_u.mean(dim=0)
    h_item_mu = H_i.mean(dim=0)

    # per-item scalar sigma: embedding norm std
    norms = torch.norm(H_i, dim=-1)      # [T, I]
    sigma_item = norms.std(dim=0)        # [I]

    # sigma 정규화(0~1)로 안정화
    smin = sigma_item.min()
    smax = sigma_item.max()
    sigma_norm = (sigma_item - smin) / (smax - smin + 1e-12)

    # 다시 eval로 복구 (안전)
    teacher.eval()

    return h_user_mu.detach(), h_item_mu.detach(), sigma_norm.detach()

def compute_weights_from_sigma(sigma_norm: torch.Tensor, alpha: float):
    """
    sigma_norm: [I] in [0,1]
    return w: [I]
    """
    w = torch.exp(-float(alpha) * sigma_norm)
    return w

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["teacher", "student", "eval", "all"], default="all")

    ap.add_argument("--split", required=True)
    ap.add_argument("--items_csv", default="data/processed/items.csv")
    ap.add_argument("--teacher_out", default="results/teacher_emb.pt")
    ap.add_argument("--student_out", default="results/student_mlp.pt")

    ap.add_argument("--teacher_model", choices=["lightgcn", "mf", "gcn"], default="lightgcn")

    # teacher common
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)

    # gcn extra
    ap.add_argument("--gcn_dropout", type=float, default=0.0)

    # ADDED: LightGCN/MF embedding dropout (MC Dropout용)
    ap.add_argument("--mc_dropout", type=float, default=0.0)

    # graph options (LightGCN/GCN에서만 사용)
    ap.add_argument("--use_item_item", choices=["OFF", "ON"], default="ON")
    ap.add_argument("--item_item_path", default="data/processed/item_item_knn_plus1.csv")
    ap.add_argument("--ii_strength", type=float, default=1.0)

    # student
    ap.add_argument("--student_epochs", type=int, default=200)
    ap.add_argument("--student_lr", type=float, default=1e-3)
    ap.add_argument("--student_bs", type=int, default=256)
    ap.add_argument("--student_hidden", type=int, default=256)
    ap.add_argument("--student_depth", type=int, default=3)
    ap.add_argument("--student_dropout", type=float, default=0.1)

    # ADDED: Uncertainty-aware weighting on student loss
    ap.add_argument("--uncertainty", choices=["OFF", "ON"], default="OFF")
    ap.add_argument("--mc_T", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=2.0)

    # eval K
    ap.add_argument("--eval_k_warm", type=int, default=10)
    ap.add_argument("--eval_k_cold", type=int, default=100)

    # eval protocol
    ap.add_argument("--eval_protocol", choices=["full", "sampled"], default="full")
    ap.add_argument("--eval_neg_samples", type=int, default=99)
    ap.add_argument("--eval_seed", type=int, default=2025)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(args.split)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = int(max(tr.user_id.max(), va.user_id.max(), te.user_id.max()))
    n_items = int(max(tr.item_id.max(), va.item_id.max(), te.item_id.max()))

    # -----------------------------
    # teacher stage
    # -----------------------------
    if args.stage in ["teacher", "all"]:
        teacher_model = args.teacher_model

        if teacher_model in ["lightgcn", "gcn"]:
            use_item_item = (args.use_item_item == "ON")
            ii_edges, ii_sims = (None, None)
            if use_item_item:
                ii_edges, ii_sims = load_item_item_knn_csv(args.item_item_path, n_items)

            A_norm, U_size, I_size = build_lightgcn_norm_adj(
                n_users, n_items, tr, device,
                use_item_item=use_item_item,
                item_item_edges=ii_edges,
                item_item_sims=ii_sims,
                ii_strength=args.ii_strength
            )

            if teacher_model == "lightgcn":
                teacher = LightGCN(
                    U_size, I_size, args.dim, args.layers, A_norm,
                    emb_dropout=args.mc_dropout
                ).to(device)
            else:
                teacher = GCNTeacher(
                    U_size, I_size, args.dim, args.layers, A_norm,
                    dropout=args.gcn_dropout
                ).to(device)

        elif teacher_model == "mf":
            U_size = int(n_users) + 1
            I_size = int(n_items) + 1
            teacher = MFTeacher(U_size, I_size, args.dim, emb_dropout=args.mc_dropout).to(device)

        else:
            raise RuntimeError(f"Unknown teacher_model={teacher_model}")

        opt = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        tr_loader = DataLoader(RatingsDS(tr), batch_size=args.bs, shuffle=True)
        va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False)
        te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False)

        best_rmse = 1e9
        best_state = None

        for ep in range(1, args.epochs + 1):
            teacher.train()
            for u, i, r in tr_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)

                h_user_b, h_item_b = teacher.compute_embeddings(mc_dropout=False)
                pred = teacher.score(h_user_b, h_item_b, u, i)

                loss = loss_fn(pred, r)
                opt.zero_grad()
                loss.backward()
                opt.step()

            teacher.eval()
            vrmse, vmae = eval_rmse_mae(teacher, va_loader, device)
            vndcg = eval_warm_ndcg_at_k(
                teacher, tr, va, n_items + 1, K=args.eval_k_warm, device=device
            )
            print(
                f"[Teacher({teacher_model}) Ep {ep:03d}] "
                f"valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@{args.eval_k_warm}={vndcg:.4f}"
            )

            if vrmse < best_rmse:
                best_rmse = vrmse
                best_state = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}

        if best_state is not None:
            teacher.load_state_dict(best_state)

        teacher.eval()
        with torch.no_grad():
            h_user_det, h_item_det = teacher.compute_embeddings(mc_dropout=False)

        # ADDED: uncertainty 계산 (MC Dropout)
        h_user_save = h_user_det
        h_item_save = h_item_det
        sigma_item = None

        if args.uncertainty == "ON":
            if args.mc_dropout <= 0 and teacher_model in ["lightgcn", "mf"]:
                print("[WARN] uncertainty=ON인데 mc_dropout=0.0 입니다. dropout을 0.1~0.2로 설정 추천.")
            hu_mu, hi_mu, sigma_norm = estimate_item_uncertainty_mc_dropout(
                teacher, T=args.mc_T, device=device
            )
            # teacher_out에는 평균 임베딩을 저장(불확실성용)
            h_user_save = hu_mu.cpu()
            h_item_save = hi_mu.cpu()
            sigma_item = sigma_norm.cpu()
            print(f"[UNC] MC Dropout done. T={args.mc_T}, sigma_norm range="
                  f"{sigma_item.min().item():.4f}~{sigma_item.max().item():.4f}")

        os.makedirs(Path(args.teacher_out).parent, exist_ok=True)
        save_dict = {
            "teacher_model": teacher_model,
            "h_user": h_user_save.detach().cpu(),
            "h_item": h_item_save.detach().cpu(),
            "n_users": n_users,
            "n_items": n_items,
            "U_size": int(h_user_save.shape[0]),
            "I_size": int(h_item_save.shape[0]),
            "dim": args.dim,
            "layers": args.layers,
            "use_item_item": args.use_item_item,
            "ii_strength": args.ii_strength,
            "gcn_dropout": args.gcn_dropout,
            "mc_dropout": args.mc_dropout,
            "uncertainty": args.uncertainty,
            "mc_T": args.mc_T,
            "alpha": args.alpha,
        }
        if sigma_item is not None:
            save_dict["sigma_item"] = sigma_item  # [I_size] in [0,1]
        torch.save(save_dict, args.teacher_out)
        print(f"[SAVE] teacher embeddings -> {args.teacher_out}")

        # teacher metrics
        cold_rmse, cold_mae = eval_rmse_mae(teacher, te_loader, device)

        scorer_t = make_scorer(h_user_det, h_item_det)  # score는 deterministic 기준(학습 모델 자체)
        if args.eval_protocol == "sampled":
            cold_recallK, cold_ndcgK = eval_sampled_recall_ndcg_at_k(
                scorer_t, tr, te, n_items + 1,
                K=args.eval_k_cold,
                n_neg=args.eval_neg_samples,
                seed=args.eval_seed + 1,
                device=device,
            )
            print(
                f"[Teacher({teacher_model}) COLD|SAMPLED neg={args.eval_neg_samples}] "
                f"RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | "
                f"Recall@{args.eval_k_cold}={cold_recallK:.6f} NDCG@{args.eval_k_cold}={cold_ndcgK:.6f}"
            )
        else:
            cold_recallK, cold_ndcgK = eval_cold_recall_ndcg_at_k(
                teacher, tr, te, n_items + 1, K=args.eval_k_cold, device=device
            )
            print(
                f"[Teacher({teacher_model}) COLD] "
                f"RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | "
                f"Recall@{args.eval_k_cold}={cold_recallK:.6f} NDCG@{args.eval_k_cold}={cold_ndcgK:.6f}"
            )

    # -----------------------------
    # student stage
    # -----------------------------
    if args.stage in ["student", "all"]:
        ckpt = torch.load(args.teacher_out, map_location="cpu")
        h_item_teacher = ckpt["h_item"]  # [I_size, dim]
        dim = h_item_teacher.shape[1]
        n_items_ckpt = int(ckpt["n_items"])

        # ADDED: sigma 로드(없으면 None)
        sigma_item = ckpt.get("sigma_item", None)  # [I_size] in [0,1] or None
        if args.uncertainty == "ON" and sigma_item is None:
            print("[WARN] uncertainty=ON인데 teacher_out에 sigma_item이 없습니다. teacher stage를 uncertainty=ON으로 다시 저장하세요.")

        feat_np, D = load_item_genre_features(args.items_csv, n_items_ckpt)
        feat = torch.from_numpy(feat_np).float()

        warm_items = sorted(set(tr["item_id"].astype(int).unique().tolist()))
        warm_items = [it for it in warm_items if 1 <= it <= n_items_ckpt]

        X = feat[warm_items]             # [W, D]
        Y = h_item_teacher[warm_items]   # [W, dim]

        # ADDED: warm item별 weight
        Ww = None
        if args.uncertainty == "ON" and sigma_item is not None:
            w_all = compute_weights_from_sigma(sigma_item.float(), alpha=args.alpha)  # [I]
            Ww = w_all[warm_items].float()  # [W]
            # 안전장치: 너무 작아지는 weight 방지(선택)
            Ww = torch.clamp(Ww, min=0.1, max=1.0)

        if Ww is None:
            ds = torch.utils.data.TensorDataset(X, Y)
        else:
            ds = torch.utils.data.TensorDataset(X, Y, Ww)

        dl = DataLoader(ds, batch_size=args.student_bs, shuffle=True)

        student = ContentMLP(
            in_dim=D, out_dim=dim,
            hidden=args.student_hidden,
            depth=args.student_depth,
            dropout=args.student_dropout
        ).to(device)

        opt = torch.optim.Adam(student.parameters(), lr=args.student_lr, weight_decay=1e-5)

        for ep in range(1, args.student_epochs + 1):
            student.train()
            losses = []

            for batch in dl:
                if Ww is None:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch

                xb, yb = xb.to(device), yb.to(device)
                pred = student(xb)

                # CHANGED: uncertainty=ON이면 weighted MSE
                if wb is None:
                    loss = torch.mean((pred - yb) ** 2)
                else:
                    wb = wb.to(device)  # [B]
                    mse_per = ((pred - yb) ** 2).mean(dim=1)  # [B]
                    loss = (wb * mse_per).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            if ep % 20 == 0 or ep == 1:
                print(f"[Student Ep {ep:03d}] MSE={np.mean(losses):.6f}")

        os.makedirs(Path(args.student_out).parent, exist_ok=True)
        torch.save({
            "state_dict": student.state_dict(),
            "in_dim": D,
            "out_dim": dim,
            "hidden": args.student_hidden,
            "depth": args.student_depth,
            "dropout": args.student_dropout,
            "uncertainty": args.uncertainty,
            "alpha": args.alpha,
        }, args.student_out)
        print(f"[SAVE] student model -> {args.student_out}")

    # -----------------------------
    # eval stage (replace cold item embeddings)
    # -----------------------------
    if args.stage in ["eval", "all"]:
        ckpt = torch.load(args.teacher_out, map_location=device)
        h_user = ckpt["h_user"].to(device)
        h_item = ckpt["h_item"].to(device)
        n_items_ckpt = int(ckpt["n_items"])

        feat_np, D = load_item_genre_features(args.items_csv, n_items_ckpt)
        feat = torch.from_numpy(feat_np).float().to(device)

        s_ckpt = torch.load(args.student_out, map_location="cpu")
        student = ContentMLP(
            in_dim=s_ckpt["in_dim"],
            out_dim=s_ckpt["out_dim"],
            hidden=s_ckpt["hidden"],
            depth=s_ckpt["depth"],
            dropout=s_ckpt["dropout"],
        ).to(device)
        student.load_state_dict(s_ckpt["state_dict"])
        student.eval()

        tr_items_set = set(tr["item_id"].astype(int).unique().tolist())
        te_items_set = set(te["item_id"].astype(int).unique().tolist())
        cold_items = sorted(list(te_items_set - tr_items_set))

        with torch.no_grad():
            warm_items = sorted([it for it in tr_items_set if 1 <= it <= n_items_ckpt])

            warm_norm = h_item[warm_items].norm(dim=1)
            print("[NORM] warm item norm mean/std:", warm_norm.mean().item(), warm_norm.std().item())

            xi = feat[cold_items]
            pred_raw = student(xi)
            pred_raw_norm = pred_raw.norm(dim=1)
            print("[NORM] student pred RAW norm mean/std:", pred_raw_norm.mean().item(), pred_raw_norm.std().item())

            user_norm = h_user[1:].norm(dim=1)
            print("[NORM] user norm mean/std:", user_norm.mean().item(), user_norm.std().item())

        h_item_final = h_item.clone()
        with torch.no_grad():
            xi = feat[cold_items]
            pred_raw = student(xi)
            h_item_final[cold_items] = pred_raw

        scorer = make_scorer(h_user, h_item_final)

        if args.eval_protocol == "sampled":
            warm_recallK, warm_ndcgK = eval_sampled_recall_ndcg_at_k(
                scorer, tr, va, n_items_ckpt + 1,
                K=args.eval_k_warm,
                n_neg=args.eval_neg_samples,
                seed=args.eval_seed,
                device=device,
            )
            cold_recallK, cold_ndcgK = eval_sampled_recall_ndcg_at_k(
                scorer, tr, te, n_items_ckpt + 1,
                K=args.eval_k_cold,
                n_neg=args.eval_neg_samples,
                seed=args.eval_seed + 1,
                device=device,
            )
            print(f"[EVAL Teacher+Student|SAMPLED neg={args.eval_neg_samples}] "
                  f"Warm Recall@{args.eval_k_warm}={warm_recallK:.6f} NDCG@{args.eval_k_warm}={warm_ndcgK:.6f} "
                  f"| Cold Recall@{args.eval_k_cold}={cold_recallK:.6f} NDCG@{args.eval_k_cold}={cold_ndcgK:.6f}")
        else:
            warm_ndcgK = eval_warm_ndcg_at_k(
                scorer, tr, va, n_items_ckpt + 1, K=args.eval_k_warm, device=device
            )
            cold_recallK, cold_ndcgK = eval_cold_recall_ndcg_at_k(
                scorer, tr, te, n_items_ckpt + 1, K=args.eval_k_cold, device=device
            )
            print(f"[EVAL Teacher+Student] Warm NDCG@{args.eval_k_warm}={warm_ndcgK:.4f} "
                  f"| Cold Recall@{args.eval_k_cold}={cold_recallK:.6f} NDCG@{args.eval_k_cold}={cold_ndcgK:.6f}")

if __name__ == "__main__":
    main()