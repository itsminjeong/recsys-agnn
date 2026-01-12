# run_gcn_improved.py
# GCN (개선판): 평점기반 edge weight + 속성게이트 + LayerNorm/Dropout/Residual + [1,5] clamp
# 평가: Warm(RMSE/MAE/NDCG@10), Cold(RMSE/MAE/Recall@10/NDCG@10)

import argparse, os, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops

from utils.metrics import (
    eval_rmse_mae,
    eval_warm_ndcg_at_k,
    eval_cold_recall_ndcg_at_k,
)

# -----------------------------
# Dataset (u,i,r)
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
# 속성 로더 (users.csv, items.csv)
# -----------------------------
def load_attrs(processed_dir: Path, n_users: int, n_items: int, device: str):
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")

    # user attrs: age(min-max) + gender(F/M) + occupation one-hot
    age = users["age"].astype(float).values
    age = ((age - np.nanmin(age)) / (np.nanmax(age) - np.nanmin(age) + 1e-8)).reshape(-1, 1).astype(np.float32)

    g_vocab = ["F", "M"]
    g_map = {g:i for i,g in enumerate(g_vocab)}
    g_oh = np.zeros((len(users), len(g_vocab)), dtype=np.float32)
    genders = users["gender"].fillna("U").astype(str).values
    for r, g in enumerate(genders):
        if g in g_map: g_oh[r, g_map[g]] = 1.0

    occ_vals = users["occupation"].fillna("other").astype(str).values
    occ_vocab = sorted(pd.Series(occ_vals).unique().tolist())
    occ_map = {o:i for i,o in enumerate(occ_vocab)}
    occ_oh = np.zeros((len(users), len(occ_vocab)), dtype=np.float32)
    for r, o in enumerate(occ_vals):
        occ_oh[r, occ_map[o]] = 1.0

    user_attr_full = np.concatenate([age, g_oh, occ_oh], axis=1).astype(np.float32)

    # items: genre_* one-hot
    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    item_attr_full = items[genre_cols].astype(np.float32).values

    # id는 0..n_users-1 / 0..n_items-1 가정 (전처리단계에서 mapping됨)
    user_attr = torch.from_numpy(user_attr_full[:n_users]).to(device)
    item_attr = torch.from_numpy(item_attr_full[:n_items]).to(device)

    return user_attr, item_attr  # [U,Du], [I,Di]

# -----------------------------
# GCN Improved
# -----------------------------
class GCNImproved(nn.Module):
    def __init__(self, n_users, n_items, dim=64, hidden=64, dropout=0.2,
                 user_attr_dim=0, item_attr_dim=0, use_attrs=False, layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.use_attrs = use_attrs
        self.layers = layers

        self.user_id_emb = nn.Embedding(n_users, dim)
        self.item_id_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_id_emb.weight, std=0.01)
        nn.init.normal_(self.item_id_emb.weight, std=0.01)

        if use_attrs:
            self.user_attr_fc = nn.Linear(user_attr_dim, dim)
            self.item_attr_fc = nn.Linear(item_attr_dim, dim)
            nn.init.xavier_uniform_(self.user_attr_fc.weight); nn.init.zeros_(self.user_attr_fc.bias)
            nn.init.xavier_uniform_(self.item_attr_fc.weight); nn.init.zeros_(self.item_attr_fc.bias)
            self.gate_u = nn.Parameter(torch.zeros(dim))
            self.gate_i = nn.Parameter(torch.zeros(dim))

        self.convs = nn.ModuleList([GCNConv(dim, hidden)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden)])
        for _ in range(layers-1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.proj_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

        # 평점 예측 바이어스 & 글로벌 평균
        self.bu = nn.Embedding(n_users, 1); nn.init.zeros_(self.bu.weight)
        self.bi = nn.Embedding(n_items, 1); nn.init.zeros_(self.bi.weight)
        self.mu = nn.Parameter(torch.zeros(1))

    def fuse_with_attrs(self, u_id, i_id, user_attr=None, item_attr=None):
        U = self.user_id_emb.weight
        I = self.item_id_emb.weight
        if self.use_attrs:
            u_attr_e = self.user_attr_fc(user_attr)  # [U,dim]
            i_attr_e = self.item_attr_fc(item_attr)  # [I,dim]
            # gate (per-dim)
            gu = torch.sigmoid(self.gate_u)
            gi = torch.sigmoid(self.gate_i)
            U = U + gu * u_attr_e
            I = I + gi * i_attr_e
        X = torch.cat([U, I], dim=0)  # [U+I, dim]
        return X

    def propagate(self, x, edge_index, edge_weight):
        h = x
        for l, (conv, ln) in enumerate(zip(self.convs, self.norms)):
            h_in = h
            h = conv(h, edge_index, edge_weight)
            h = ln(h)
            h = torch.relu(h)
            h = self.dropout(h)
            # Residual (차원 동일 가정)
            h = h + h_in
        h = self.proj_out(h)  # back to dim
        return h[:self.n_users], h[self.n_users:]  # (h_user, h_item)

    def forward(self, users, items, edge_index, edge_weight, user_attr=None, item_attr=None):
        # 전체 임베딩 전파
        x0 = self.fuse_with_attrs(self.user_id_emb, self.item_id_emb, user_attr, item_attr)
        h_user, h_item = self.propagate(x0, edge_index, edge_weight)
        # 점수
        s  = (h_user[users] * h_item[items]).sum(-1)
        s += self.bu(users).squeeze(-1) + self.bi(items).squeeze(-1) + self.mu
        return torch.clamp(s, 1.0, 5.0)

    @torch.no_grad()
    def predict(self, users, items, h_user, h_item):
        s  = (h_user[users] * h_item[items]).sum(-1)
        s += self.bu(users).squeeze(-1) + self.bi(items).squeeze(-1) + self.mu
        return torch.clamp(s, 1.0, 5.0)

# -----------------------------
# 그래프 구성 (edge_index/weight)
# -----------------------------
def build_graph(tr: pd.DataFrame, n_users: int, n_items: int, device: str):
    # bipartite indices: user -> [0..U-1], item -> [U..U+I-1]
    u = torch.tensor(tr["user_id"].values, dtype=torch.long)
    i = torch.tensor(tr["item_id"].values, dtype=torch.long)
    r = torch.tensor(tr["rating"].values,  dtype=torch.float32)

    src = torch.cat([u,        i + n_users], dim=0)
    dst = torch.cat([i + n_users, u       ], dim=0)
    edge_index = torch.stack([src, dst], dim=0)  # [2, 2E]

    # rating-based weights → 정규화(평균 3.0 기준), 최소값 보정
    # 간단히: w = 0.5 + (r-3)/4 ∈ [0.25, 1.0] 정도
    w = 0.5 + (r - 3.0) / 4.0
    w = torch.clamp(w, 0.1, 1.0)
    edge_weight = torch.cat([w, w], dim=0)

    # self-loop 추가 (PyG 버전 호환: positional args 사용)
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=n_users + n_items
    )
    return edge_index.to(device), edge_weight.to(device)

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/seed42_u0.1_i0.0")
    ap.add_argument("--attrs", choices=["OFF","ON"], default="ON")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--gcn_hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(args.split)

    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = max(tr.user_id.max(), va.user_id.max(), te.user_id.max()) + 1
    n_items = max(tr.item_id.max(), va.item_id.max(), te.item_id.max()) + 1

    tr_loader = DataLoader(RatingsDS(tr), batch_size=args.bs, shuffle=True, num_workers=0)
    va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)
    te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)

    # 그래프
    edge_index, edge_weight = build_graph(tr, n_users, n_items, device)

    # 속성
    use_attrs = (args.attrs == "ON")
    if use_attrs:
        user_attr, item_attr = load_attrs(Path("data/processed"), n_users, n_items, device)
        uad, iad = user_attr.shape[1], item_attr.shape[1]
    else:
        user_attr, item_attr = None, None
        uad, iad = 0, 0

    # 모델
    model = GCNImproved(
        n_users=n_users, n_items=n_items,
        dim=args.dim, hidden=args.gcn_hidden, dropout=args.dropout,
        user_attr_dim=uad, item_attr_dim=iad, use_attrs=use_attrs,
        layers=args.layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    loss_fn = nn.MSELoss()

    best = (1e9, 1e9, -1.0)  # (rmse, mae, ndcg)
    best_ep = 0

    # 학습
    for ep in range(1, args.epochs+1):
        model.train()
        for u, i, r in tr_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i, edge_index, edge_weight, user_attr, item_attr)
            loss = loss_fn(pred, r)
            opt.zero_grad(); loss.backward(); opt.step()

        # warm RMSE/MAE
        def predict_fn(u, i):
            # 에폭 끝에서 캐시 전파(한 번만)
            return model(u, i, edge_index, edge_weight, user_attr, item_attr)
        vrmse, vmae = eval_rmse_mae(model, va_loader, device, predict_fn=predict_fn)

        # warm NDCG@10 (캐시 사용)
        model.eval()
        with torch.no_grad():
            x0 = model.fuse_with_attrs(model.user_id_emb, model.item_id_emb, user_attr, item_attr)
            h_user, h_item = model.propagate(x0, edge_index, edge_weight)
        vndcg = eval_warm_ndcg_at_k(
            model, tr, va, n_items, K=10, device=device,
            gcn_cache=(h_user, h_item)
        )
        print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

        if vrmse < best[0]:
            best = (vrmse, vmae, vndcg)
            best_ep = ep
            torch.save(model.state_dict(), "best_gcn_imp.pth")

    # 로드 & 최종 캐시
    model.load_state_dict(torch.load("best_gcn_imp.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        x0 = model.fuse_with_attrs(model.user_id_emb, model.item_id_emb, user_attr, item_attr)
        h_user, h_item = model.propagate(x0, edge_index, edge_weight)

    # warm(=valid) 지표 재확인(표기용)
    def predict_fn(u,i):
        return model.predict(u,i,h_user,h_item)
    vrmse, vmae = eval_rmse_mae(model, va_loader, device, predict_fn=predict_fn)
    vndcg = eval_warm_ndcg_at_k(model, tr, va, n_items, K=10, device=device, gcn_cache=(h_user,h_item))

    # cold 지표
    # RMSE/MAE(회귀)도 동일 predict_fn으로
    trmse, tmae = eval_rmse_mae(model, te_loader, device, predict_fn=predict_fn)
    recall10, ndcg10 = eval_cold_recall_ndcg_at_k(model, tr, te, n_items, K=10, device=device, gcn_cache=(h_user,h_item))

    print(f"[COLD TEST] RMSE={trmse:.4f} MAE={tmae:.4f} | Recall@10={recall10:.4f} NDCG@10={ndcg10:.4f} (best Ep {best_ep})")

    # 저장
    os.makedirs("results", exist_ok=True)
    need_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["model","attrs","split_dir","dim","layers","epochs",
                        "valid_rmse","valid_mae","valid_ndcg10",
                        "cold_rmse","cold_mae","cold_recall10","cold_ndcg10"])
        w.writerow([
            "GCN+IMP", args.attrs, str(split_dir),
            args.dim, args.layers, best_ep,
            vrmse, vmae, vndcg,
            trmse, tmae, recall10, ndcg10
        ])

if __name__ == "__main__":
    main()