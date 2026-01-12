# run_gcn.py  (교체본)
# 목표
# - GCN (attrs OFF/ON) 학습/평가
# - Warm: RMSE/MAE + NDCG@10
# - Cold: RMSE/MAE + Recall@10 + NDCG@10

import argparse, os, csv
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

# =========================
# Dataset
# =========================
class RatingsDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)
    def __len__(self): return self.r.numel()
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

# =========================
# Attributes (ON일 때만)
# =========================
def load_user_item_attrs(processed_dir: Path, n_users: int, n_items: int):
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")

    age = users["age"].astype(float).values
    age = ((age - np.nanmin(age)) / (np.nanmax(age) - np.nanmin(age) + 1e-8)).reshape(-1, 1).astype(np.float32)

    g_vocab = ["F", "M"]; g_map = {g:i for i,g in enumerate(g_vocab)}
    g_oh = np.zeros((len(users), len(g_vocab)), dtype=np.float32)
    for r, g in enumerate(users["gender"].fillna("U").astype(str).values):
        if g in g_map: g_oh[r, g_map[g]] = 1.0

    occ_vals = users["occupation"].fillna("other").astype(str).values
    occ_vocab = sorted(pd.Series(occ_vals).unique().tolist()); occ_map = {o:i for i,o in enumerate(occ_vocab)}
    occ_oh = np.zeros((len(users), len(occ_vocab)), dtype=np.float32)
    for r, o in enumerate(occ_vals): occ_oh[r, occ_map[o]] = 1.0

    user_attr = np.concatenate([age, g_oh, occ_oh], axis=1).astype(np.float32)

    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    if not genre_cols:
        raise RuntimeError("items.csv에 genre_* 컬럼이 없습니다.")
    item_attr = items[genre_cols].astype(np.float32).values

    user_attr = user_attr[:n_users]
    item_attr = item_attr[:n_items]
    return user_attr, item_attr

# =========================
# Sparse normalized adjacency
# =========================
def build_normalized_adj(n_users: int, n_items: int, train_df: pd.DataFrame, device: str):
    u = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    i = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    num_nodes = n_users + n_items

    rows = torch.cat([u, i + n_users], dim=0)
    cols = torch.cat([i + n_users, u], dim=0)
    vals = torch.ones(rows.numel(), dtype=torch.float32)

    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0), vals,
        size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    d_row = deg_inv_sqrt[rows]
    d_col = deg_inv_sqrt[cols]
    nvals = d_row * vals * d_col

    A_norm = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0), nvals,
        size=(num_nodes, num_nodes), dtype=torch.float32, device=device
    ).coalesce()
    return A_norm

# =========================
# GCN Model
# =========================
class GCNRecsys(nn.Module):
    def __init__(self, n_users, n_items, dim=64, layers=2,
                 use_attrs=False, user_attr_dim=0, item_attr_dim=0, dropout=0.0):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.dim, self.layers = dim, layers
        self.use_attrs = use_attrs

        self.Eu = nn.Embedding(n_users, dim)
        self.Ei = nn.Embedding(n_items, dim)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))

        if use_attrs:
            self.user_attr_fc = nn.Linear(user_attr_dim, dim)
            self.item_attr_fc = nn.Linear(item_attr_dim, dim)
        else:
            self.user_attr_fc = None
            self.item_attr_fc = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.Eu.weight, std=0.01)
        nn.init.normal_(self.Ei.weight, std=0.01)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        if self.user_attr_fc is not None:
            nn.init.xavier_uniform_(self.user_attr_fc.weight); nn.init.zeros_(self.user_attr_fc.bias)
        if self.item_attr_fc is not None:
            nn.init.xavier_uniform_(self.item_attr_fc.weight); nn.init.zeros_(self.item_attr_fc.bias)

        # 그래프/속성 캐시(encode용)
        self.A_norm = None
        self.ua = None
        self.ia = None

    def set_graph(self, A_norm, ua_t=None, ia_t=None):
        self.A_norm = A_norm
        self.ua = ua_t
        self.ia = ia_t

    def _initial_x(self):
        U0 = self.Eu.weight
        I0 = self.Ei.weight
        if self.use_attrs:
            U0 = U0 + self.user_attr_fc(self.ua)
            I0 = I0 + self.item_attr_fc(self.ia)
        X0 = torch.cat([U0, I0], dim=0)  # [U+I, d]
        return X0

    def _propagate(self, X):
        H = X
        for _ in range(self.layers):
            H = torch.sparse.mm(self.A_norm, H)
            H = self.dropout(H)
        U = H[:self.n_users]
        I = H[self.n_users:]
        return U, I

    # --------- 학습 시: grad 경로 포함 ----------
    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor):
        assert self.A_norm is not None, "call set_graph(A_norm, ua, ia) first."
        X0 = self._initial_x()           # grad 연결됨
        U, I = self._propagate(X0)       # grad 연결됨
        pred = (U[u_idx] * I[i_idx]).sum(-1)
        pred = pred + self.bu(u_idx).squeeze(-1) + self.bi(i_idx).squeeze(-1) + self.mu
        return pred  # 학습 중엔 clamp 하지 않음

    # --------- 평가 시: 한 번만 전파해서 캐시 ----------
    @torch.no_grad()
    def encode(self):
        assert self.A_norm is not None, "call set_graph(A_norm, ua, ia) first."
        X0 = self._initial_x()
        U, I = self._propagate(X0)
        return U, I

    @torch.no_grad()
    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor, U: torch.Tensor, I: torch.Tensor):
        pred = (U[u_idx] * I[i_idx]).sum(-1)
        pred = pred + self.bu(u_idx).squeeze(-1) + self.bi(i_idx).squeeze(-1) + self.mu
        return pred.clamp(1.0, 5.0)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/seed42_u0.1_i0.0")
    ap.add_argument("--attrs", choices=["OFF","ON"], default="OFF")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    use_attrs = (args.attrs.upper() == "ON")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_dir = Path(args.split)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = int(max(tr.user_id.max(), va.user_id.max(), te.user_id.max())) + 1
    n_items = int(max(tr.item_id.max(), va.item_id.max(), te.item_id.max())) + 1

    tr_loader = DataLoader(RatingsDS(tr), batch_size=args.bs, shuffle=True,  num_workers=0)
    va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)
    te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)

    if use_attrs:
        ua_np, ia_np = load_user_item_attrs(Path("data/processed"), n_users, n_items)
        ua_t = torch.from_numpy(ua_np).to(device)
        ia_t = torch.from_numpy(ia_np).to(device)
    else:
        ua_t, ia_t = None, None

    A_norm = build_normalized_adj(n_users, n_items, tr, device=device)

    model = GCNRecsys(
        n_users=n_users, n_items=n_items,
        dim=args.dim, layers=args.layers,
        use_attrs=use_attrs,
        user_attr_dim=(ua_t.shape[1] if use_attrs else 0),
        item_attr_dim=(ia_t.shape[1] if use_attrs else 0),
        dropout=args.dropout
    ).to(device)
    model.set_graph(A_norm, ua_t, ia_t)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best = (1e9, 1e9, -1.0)
    best_ep = 0

    for ep in range(1, args.epochs+1):
        model.train()
        for u, i, r in tr_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)          # ← 전파가 grad 경로로 포함됨
            loss = loss_fn(pred, r)
            opt.zero_grad(); loss.backward(); opt.step()

        # 평가: 에폭 끝에 한 번만 전파해서 고정 U,I 사용
        U, I = model.encode()

        vrmse, vmae = eval_rmse_mae(model, va_loader, device,
                                    predict_fn=lambda u,i: model.score(u,i,U,I))
        vndcg = eval_warm_ndcg_at_k(
            model_or_scorer=lambda u,i: model.score(u,i,U,I),
            train_df=tr, valid_df=va, n_items=n_items, K=10, device=device
        )
        print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

        if vrmse < best[0]:
            best = (vrmse, vmae, vndcg); best_ep = ep

    # 최종 평가
    U, I = model.encode()

    cold_rmse, cold_mae = eval_rmse_mae(model, te_loader, device,
                                        predict_fn=lambda u,i: model.score(u,i,U,I))
    cold_recall10, cold_ndcg10 = eval_cold_recall_ndcg_at_k(
        model_or_scorer=lambda u,i: model.score(u,i,U,I),
        train_df=tr, cold_df=te, n_items=n_items, K=10, device=device
    )
    print(f"[COLD TEST] RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | Recall@10={cold_recall10:.4f} NDCG@10={cold_ndcg10:.4f} (best Ep {best_ep})")

    os.makedirs("results", exist_ok=True)
    need_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["model","attrs","split_dir","dim","layers","epochs",
                        "valid_rmse","valid_mae","valid_ndcg10",
                        "cold_rmse","cold_mae","cold_recall10","cold_ndcg10"])
        w.writerow(["GCN", args.attrs, str(split_dir), args.dim, args.layers, args.epochs,
                    best[0], best[1], best[2],
                    cold_rmse, cold_mae, cold_recall10, cold_ndcg10])

if __name__ == "__main__":
    main()