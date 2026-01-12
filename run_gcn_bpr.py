# run_gcn_bpr.py  (FIXED)
# ------------------------------------------------------------
# 목적:
# - LightGCN 스타일의 GCN + BPR 학습으로 랭킹 지표(Recall/NDCG) 향상 확인
# - 에폭마다/배치마다 임베딩을 학습 그래프 위에서 계산(훈련 시 no_grad 금지)
# - attrs ON이면 간단한 속성 게이트(learnable alpha)로 사용자/아이템 속성 주입
#
# 입출력:
# - 입력 split 디렉토리: {split}/train.csv, valid.csv, test_cold.csv (cols: user_id,item_id,rating)
# - attrs ON일 경우 data/processed/{users.csv, items.csv} (user/item 속성)
# - 결과 행을 results/metrics_bpr.csv에 append
#
# 지표:
# - Warm: RMSE/MAE/NDCG@10
# - Cold: Recall@10/NDCG@10 (+ RMSE/MAE 참고 출력)
# ------------------------------------------------------------

import argparse, os, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.metrics import (
    eval_rmse_mae,
    eval_warm_ndcg_at_k,
    eval_cold_recall_ndcg_at_k,
)

# -----------------------------
# Dataset (for RMSE/MAE eval)
# -----------------------------
class RatingsDS(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self): return self.r.shape[0]
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

# -----------------------------
# 속성 로딩 (옵션)
# -----------------------------
def build_user_item_attr_mats(processed_dir: Path, n_users: int, n_items: int):
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")

    # user attrs: age(min-max) + gender(F/M) + occupation(one-hot)
    age = users["age"].astype(float).values
    age = ((age - np.nanmin(age)) / (np.nanmax(age) - np.nanmin(age) + 1e-8)).reshape(-1, 1).astype(np.float32)

    g_vocab = ["F", "M"]; g_map = {g:i for i,g in enumerate(g_vocab)}
    g_oh = np.zeros((len(users), len(g_vocab)), dtype=np.float32)
    for r, g in enumerate(users["gender"].fillna("U").astype(str).values):
        if g in g_map: g_oh[r, g_map[g]] = 1.0

    occ_vals = users["occupation"].fillna("other").astype(str).values
    occ_vocab = sorted(pd.Series(occ_vals).unique().tolist())
    occ_map = {o:i for i,o in enumerate(occ_vocab)}
    occ_oh = np.zeros((len(users), len(occ_vocab)), dtype=np.float32)
    for r, o in enumerate(occ_vals): occ_oh[r, occ_map[o]] = 1.0

    user_attr = np.concatenate([age, g_oh, occ_oh], axis=1).astype(np.float32)

    uids = users["user_id"].astype(int).values
    max_uid = uids.max()
    user_attr_mat = np.zeros((max(max_uid+1, n_users), user_attr.shape[1]), dtype=np.float32)
    user_attr_mat[uids] = user_attr
    user_attr_mat = user_attr_mat[:n_users]

    # item attrs: genre_* one-hot
    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    if not genre_cols:
        raise RuntimeError("items.csv에서 genre_* 컬럼을 찾지 못했습니다.")
    item_attr = items[genre_cols].astype(np.float32).values

    iids = items["item_id"].astype(int).values
    max_iid = iids.max()
    item_attr_mat = np.zeros((max(max_iid+1, n_items), item_attr.shape[1]), dtype=np.float32)
    item_attr_mat[iids] = item_attr
    item_attr_mat = item_attr_mat[:n_items]

    return user_attr_mat, item_attr_mat

# -----------------------------
# LightGCN-like Encoder
# -----------------------------
class LightGCNEncoder(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=2,
                 use_attrs=False, user_attr_dim=0, item_attr_dim=0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.use_attrs = use_attrs

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        if use_attrs:
            self.user_attr_fc = nn.Linear(user_attr_dim, dim)
            self.item_attr_fc = nn.Linear(item_attr_dim, dim)
            nn.init.xavier_uniform_(self.user_attr_fc.weight); nn.init.zeros_(self.user_attr_fc.bias)
            nn.init.xavier_uniform_(self.item_attr_fc.weight); nn.init.zeros_(self.item_attr_fc.bias)
            self.gate_u = nn.Parameter(torch.tensor(0.0))
            self.gate_i = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("edge_index", None)   # [2, E]
        self.register_buffer("norm", None)         # [E]

    def build_graph(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        device = user_ids.device
        u = user_ids
        v = item_ids + self.n_users
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)  # undirected
        N = self.n_users + self.n_items
        deg = torch.bincount(edge_index.view(-1), minlength=N).float().clamp(min=1.0).to(device)
        d_inv_sqrt = deg.pow(-0.5)
        src, dst = edge_index[0], edge_index[1]
        norm = d_inv_sqrt[src] * d_inv_sqrt[dst]
        self.edge_index = edge_index
        self.norm = norm

    def propagate(self, x):
        src, dst = self.edge_index[0], self.edge_index[1]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] * self.norm.unsqueeze(-1))
        return out

    def forward(self, users_all: torch.Tensor, items_all: torch.Tensor,
                user_attr_mat: torch.Tensor = None, item_attr_mat: torch.Tensor = None):
        U0 = self.user_emb(users_all)     # [U, d]
        I0 = self.item_emb(items_all)     # [I, d]

        if self.use_attrs and (user_attr_mat is not None) and (item_attr_mat is not None):
            Ua = self.user_attr_fc(user_attr_mat.float())
            Ia = self.item_attr_fc(item_attr_mat.float())
            U0 = U0 + torch.sigmoid(self.gate_u) * Ua
            I0 = I0 + torch.sigmoid(self.gate_i) * Ia

        X0 = torch.cat([U0, I0], dim=0)  # [U+I, d]
        X = X0
        outs = [X0]
        for _ in range(self.n_layers):
            X = self.propagate(X)
            outs.append(X)
        X_final = torch.mean(torch.stack(outs, dim=0), dim=0)
        h_user = X_final[:self.n_users]
        h_item = X_final[self.n_users:]
        return h_user, h_item

    @staticmethod
    def predict(u: torch.Tensor, i: torch.Tensor,
                h_user: torch.Tensor, h_item: torch.Tensor) -> torch.Tensor:
        return (h_user[u] * h_item[i]).sum(dim=-1)

# -----------------------------
# BPR utilities
# -----------------------------
def bpr_loss(u, pos_i, neg_i, h_user, h_item, l2_reg=1e-4):
    pos = (h_user[u] * h_item[pos_i]).sum(-1)
    neg = (h_user[u] * h_item[neg_i]).sum(-1)
    loss = -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()
    loss = loss + l2_reg * (h_user[u].pow(2).mean() + h_item[pos_i].pow(2).mean() + h_item[neg_i].pow(2).mean())
    return loss

def build_train_pos_dict(df: pd.DataFrame):
    d = {}
    for u, i in zip(df["user_id"].values, df["item_id"].values):
        d.setdefault(int(u), set()).add(int(i))
    return d

def sample_neg_items(u_batch: np.ndarray, n_items: int, train_pos_dict, rng: np.random.RandomState):
    neg = []
    for u in u_batch:
        tried = 0
        while True:
            j = int(rng.randint(0, n_items))
            if j not in train_pos_dict.get(int(u), set()):
                neg.append(j); break
            tried += 1
            if tried > 100:
                neg.append(j); break
    return np.array(neg, dtype=np.int64)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--attrs", choices=["OFF", "ON"], default="OFF")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="results/metrics_bpr.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_dir = Path(args.split)

    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = max(tr.user_id.max(), va.user_id.max(), te.user_id.max()) + 1
    n_items = max(tr.item_id.max(), va.item_id.max(), te.item_id.max()) + 1

    va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)

    use_attrs = (args.attrs == "ON")
    user_attr_mat_t = item_attr_mat_t = None
    user_attr_dim = item_attr_dim = 0
    if use_attrs:
        ua_np, ia_np = build_user_item_attr_mats(Path("data/processed"), n_users, n_items)
        user_attr_mat_t = torch.from_numpy(ua_np).to(device)   # << 한 번만 GPU로 올림
        item_attr_mat_t = torch.from_numpy(ia_np).to(device)
        user_attr_dim = ua_np.shape[1]; item_attr_dim = ia_np.shape[1]

    model = LightGCNEncoder(
        n_users=n_users, n_items=n_items,
        dim=args.dim, n_layers=args.layers,
        use_attrs=use_attrs,
        user_attr_dim=user_attr_dim, item_attr_dim=item_attr_dim
    ).to(device)

    model.build_graph(
        torch.tensor(tr["user_id"].values, dtype=torch.long, device=device),
        torch.tensor(tr["item_id"].values, dtype=torch.long, device=device),
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)
    train_pos_dict = build_train_pos_dict(tr)

    train_ui = torch.utils.data.TensorDataset(
        torch.tensor(tr["user_id"].values, dtype=torch.long),
        torch.tensor(tr["item_id"].values, dtype=torch.long)
    )
    train_loader = DataLoader(train_ui, batch_size=args.bs, shuffle=True, drop_last=False)

    best_v = (-1.0, 1e9, 1e9)  # (NDCG, RMSE, MAE)
    best_ep = 0

    # -------- Training --------
    for ep in range(1, args.epochs + 1):
        model.train()
        for ub, ib in train_loader:
            ub = ub.to(device); ib = ib.to(device)

            # (1) 임베딩 계산 (학습 중: no_grad 금지!)
            all_users = torch.arange(n_users, device=device, dtype=torch.long)
            all_items = torch.arange(n_items, device=device, dtype=torch.long)
            h_user, h_item = model(
                all_users, all_items,
                user_attr_mat=user_attr_mat_t if use_attrs else None,
                item_attr_mat=item_attr_mat_t if use_attrs else None,
            )

            # (2) 네거티브 샘플링
            neg_np = sample_neg_items(ub.detach().cpu().numpy(), n_items, train_pos_dict, rng)
            jb = torch.tensor(neg_np, dtype=torch.long, device=device)

            # (3) BPR loss (grad 흐름 O)
            loss = bpr_loss(ub, ib, jb, h_user, h_item, l2_reg=1e-4)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        # ----- 에폭 종료: 평가 -----
        model.eval()
        with torch.no_grad():
            all_users = torch.arange(n_users, device=device, dtype=torch.long)
            all_items = torch.arange(n_items, device=device, dtype=torch.long)
            h_user, h_item = model(
                all_users, all_items,
                user_attr_mat=user_attr_mat_t if use_attrs else None,
                item_attr_mat=item_attr_mat_t if use_attrs else None,
            )

            vrmse, vmae = eval_rmse_mae(
                model, va_loader, device,
                predict_fn=lambda u, i: LightGCNEncoder.predict(u, i, h_user, h_item)
            )
            vndcg = eval_warm_ndcg_at_k(
                model_or_scorer=model,
                train_df=tr, valid_df=va,
                n_items=n_items, K=10, device=device,
                gcn_cache=(h_user, h_item)
            )

        print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

        if (vndcg > best_v[0]) or (vndcg == best_v[0] and vrmse < best_v[1]):
            best_v = (vndcg, vrmse, vmae)
            best_ep = ep

    # -------- Cold Eval --------
    te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)
    with torch.no_grad():
        all_users = torch.arange(n_users, device=device, dtype=torch.long)
        all_items = torch.arange(n_items, device=device, dtype=torch.long)
        h_user, h_item = model(
            all_users, all_items,
            user_attr_mat=user_attr_mat_t if use_attrs else None,
            item_attr_mat=item_attr_mat_t if use_attrs else None,
        )

        cold_rmse, cold_mae = eval_rmse_mae(
            model, te_loader, device,
            predict_fn=lambda u, i: LightGCNEncoder.predict(u, i, h_user, h_item)
        )
        cold_recall10, cold_ndcg10 = eval_cold_recall_ndcg_at_k(
            model_or_scorer=model,
            train_df=tr, cold_df=te,
            n_items=n_items, K=10, device=device,
            gcn_cache=(h_user, h_item)
        )

    print(f"[COLD TEST] RMSE={cold_rmse:.4f} MAE={cold_mae:.4f} | Recall@10={cold_recall10:.4f} NDCG@10={cold_ndcg10:.4f} (best Ep {best_ep})")

    os.makedirs("results", exist_ok=True)
    need_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "model","attrs","split_dir","dim","layers","epochs",
                "valid_rmse","valid_mae","valid_ndcg10",
                "cold_rmse","cold_mae","cold_recall10","cold_ndcg10",
            ])
        w.writerow([
            "GCN+BPR", args.attrs, str(split_dir), args.dim, args.layers, best_ep,
            best_v[1], best_v[2], best_v[0],
            cold_rmse, cold_mae, cold_recall10, cold_ndcg10
        ])

if __name__ == "__main__":
    main()