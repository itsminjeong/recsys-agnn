# run_mf_attr.py
# 목적:
# - MF(+Attributes) 학습
# - Warm(valid): RMSE, MAE, NDCG@10
# - Cold(test_cold): RMSE, MAE, Recall@10, NDCG@10 (user-cold / item-cold 모두 지원)
# - 결과를 results/metrics.csv에 누적 저장
#
# 핵심 포인트:
# - 속성 매트릭스를 Torch Tensor로 device에 올려 인덱싱
# - 콜드 평가 시 split 폴더 안의 cold_users.csv / cold_items.csv 존재 여부로 te를 필터링

from utils.metrics import (
    eval_rmse_mae,
    eval_warm_ndcg_at_k,
    eval_cold_recall_ndcg_at_k,
)
import argparse, os, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 속성 로딩/벡터화
# -----------------------------
def build_user_item_attr_mats(processed_dir: Path):
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")

    # --- Users ---
    user_ids = users["user_id"].unique().tolist()
    uid2idx = {u: i for i, u in enumerate(user_ids)}

    # age [0,1]
    age = users["age"].astype(float).values
    age = ((age - np.nanmin(age)) / (np.nanmax(age) - np.nanmin(age) + 1e-8)).reshape(-1, 1).astype(np.float32)

    # gender one-hot
    genders = users["gender"].fillna("U").astype(str).values
    g_vocab = ["F", "M"]
    g_map = {g: i for i, g in enumerate(g_vocab)}
    g_oh = np.zeros((len(users), len(g_vocab)), dtype=np.float32)
    for r, g in enumerate(genders):
        if g in g_map:
            g_oh[r, g_map[g]] = 1.0

    # occupation one-hot
    occ_vals = users["occupation"].fillna("other").astype(str).values
    occ_vocab = sorted(pd.Series(occ_vals).unique().tolist())
    occ_map = {o: i for i, o in enumerate(occ_vocab)}
    occ_oh = np.zeros((len(users), len(occ_vocab)), dtype=np.float32)
    for r, o in enumerate(occ_vals):
        occ_oh[r, occ_map[o]] = 1.0

    user_attr = np.concatenate([age, g_oh, occ_oh], axis=1).astype(np.float32)

    user_attr_mat = np.zeros((len(user_ids), user_attr.shape[1]), dtype=np.float32)
    for r in range(len(users)):
        u_orig = int(users.iloc[r]["user_id"])
        user_attr_mat[uid2idx[u_orig]] = user_attr[r]

    # --- Items ---
    item_ids = items["item_id"].unique().tolist()
    iid2idx = {it: i for i, it in enumerate(item_ids)}
    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    if not genre_cols:
        raise RuntimeError("items.csv에 genre_* 컬럼이 없습니다.")
    item_attr = items[genre_cols].astype(np.float32).values

    item_attr_mat = np.zeros((len(item_ids), item_attr.shape[1]), dtype=np.float32)
    for r in range(len(items)):
        it_orig = int(items.iloc[r]["item_id"])
        item_attr_mat[iid2idx[it_orig]] = item_attr[r]

    return user_attr_mat, item_attr_mat

# -----------------------------
# Dataset (속성 인덱싱 포함)
# -----------------------------
class RatingsWithAttrsDS(Dataset):
    def __init__(self, df: pd.DataFrame, user_attr_t: torch.Tensor, item_attr_t: torch.Tensor):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.r = torch.tensor(df["rating"].values, dtype=torch.float32)
        self.user_attr_t = user_attr_t
        self.item_attr_t = item_attr_t

    def __len__(self): return self.r.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx]; i = self.i[idx]; r = self.r[idx]
        ua = self.user_attr_t[u]
        ia = self.item_attr_t[i]
        return u, i, ua, ia, r

# -----------------------------
# Model (MF + Attr)
# -----------------------------
class MFWithAttrs(nn.Module):
    def __init__(self, n_users, n_items, user_attr_dim, item_attr_dim, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_attr_fc = nn.Linear(user_attr_dim, dim)
        self.item_attr_fc = nn.Linear(item_attr_dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.user_attr_fc.weight); nn.init.zeros_(self.user_attr_fc.bias)
        nn.init.xavier_uniform_(self.item_attr_fc.weight); nn.init.zeros_(self.item_attr_fc.bias)

    def forward(self, u, i, ua, ia):
        u_e = self.user_emb(u) + self.user_attr_fc(ua)
        i_e = self.item_emb(i) + self.item_attr_fc(ia)
        x = torch.cat([u_e, i_e], dim=-1)
        return self.head(x).squeeze(-1)

# -----------------------------
# 평가 헬퍼 (RMSE/MAE for Attr)
# -----------------------------
@torch.no_grad()
def eval_rmse_mae_attr(model, loader, device):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for u, i, ua, ia, r in loader:
        u, i, ua, ia, r = u.to(device), i.to(device), ua.to(device), ia.to(device), r.to(device)
        p = model(u, i, ua, ia)
        se += torch.sum((p - r) ** 2).item()
        ae += torch.sum(torch.abs(p - r)).item()
        n  += r.numel()
    return (se / n) ** 0.5, (ae / n)

# -----------------------------
# Scorer for ranking (Attr)
# -----------------------------
class AttrScorer:
    def __init__(self, model: MFWithAttrs, ua_dev: torch.Tensor, ia_dev: torch.Tensor):
        self.model = model
        self.ua = ua_dev
        self.ia = ia_dev

    @torch.no_grad()
    def __call__(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.model(users, items, self.ua[users], self.ia[items])

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/seed42_u0.1_i0.0")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로드
    split_dir = Path(args.split)
    tr = pd.read_csv(split_dir / "train.csv")
    va = pd.read_csv(split_dir / "valid.csv")
    te = pd.read_csv(split_dir / "test_cold.csv")

    n_users = int(max(tr.user_id.max(), va.user_id.max(), te.user_id.max()) + 1)
    n_items = int(max(tr.item_id.max(), va.item_id.max(), te.item_id.max()) + 1)

    # 속성 매트릭스 (numpy -> torch)
    ua_np, ia_np = build_user_item_attr_mats(Path("data/processed"))
    ua_np, ia_np = ua_np[:n_users], ia_np[:n_items]
    ua_dev = torch.from_numpy(ua_np).to(device)  # [n_users, Du]
    ia_dev = torch.from_numpy(ia_np).to(device)  # [n_items, Di]

    # DataLoaders
    tr_loader = DataLoader(RatingsWithAttrsDS(tr, ua_dev, ia_dev), batch_size=args.bs, shuffle=True,  num_workers=0)
    va_loader = DataLoader(RatingsWithAttrsDS(va, ua_dev, ia_dev), batch_size=args.bs, shuffle=False, num_workers=0)

    # ---- Cold eval용 te를 cold 유형에 따라 필터링 ----
    cold_users_p = split_dir / "cold_users.csv"
    cold_items_p = split_dir / "cold_items.csv"
    te_eval = te.copy()

    if cold_items_p.exists():
        # item-cold: test를 cold_items에 해당하는 행만 남김
        cold_items = set(pd.read_csv(cold_items_p, header=None)[0].astype(int).tolist())
        te_eval = te_eval[te_eval["item"].isin(cold_items)]
    elif cold_users_p.exists():
        # user-cold: test를 cold_users에 해당하는 행만 남김
        cold_users = set(pd.read_csv(cold_users_p, header=None)[0].astype(int).tolist())
        te_eval = te_eval[te_eval["user"].isin(cold_users)]

    te_loader = DataLoader(RatingsWithAttrsDS(te_eval, ua_dev, ia_dev), batch_size=args.bs, shuffle=False, num_workers=0)

    # Model
    model = MFWithAttrs(n_users, n_items, ua_dev.shape[1], ia_dev.shape[1], dim=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # 랭킹 scorer
    scorer = AttrScorer(model, ua_dev, ia_dev)

    best = (1e9, 1e9, -1.0)
    best_ep = 0

    # Train
    for ep in range(1, args.epochs + 1):
        model.train()
        for u, i, ua, ia, r in tr_loader:
            u, i, ua, ia, r = u.to(device), i.to(device), ua.to(device), ia.to(device), r.to(device)
            pred = model(u, i, ua, ia)
            loss = loss_fn(pred, r)
            opt.zero_grad(); loss.backward(); opt.step()

        # Warm eval
        vrmse, vmae = eval_rmse_mae_attr(model, va_loader, device)
        vndcg = eval_warm_ndcg_at_k(scorer, tr, va, n_items, K=10, device=device)
        print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

        if vrmse < best[0]:
            best = (vrmse, vmae, vndcg)
            best_ep = ep

    # Cold eval (필터링된 te_eval 사용!)
    trmse, tmae = eval_rmse_mae_attr(model, te_loader, device)
    recall10, ndcg10 = eval_cold_recall_ndcg_at_k(scorer, tr, te_eval, n_items, K=10, device=device)
    print(f"[COLD TEST] RMSE={trmse:.4f} MAE={tmae:.4f} | Recall@10={recall10:.4f} NDCG@10={ndcg10:.4f} (best Ep {best_ep})")

    # Save
    os.makedirs("results", exist_ok=True)
    need_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["model","attrs","split_dir","dim","epochs",
                        "valid_rmse","valid_mae","valid_ndcg10",
                        "cold_rmse","cold_mae","cold_recall10","cold_ndcg10"])
        w.writerow(["MF","ON",str(split_dir),args.dim,best_ep,
                    best[0],best[1],best[2],
                    trmse,tmae,recall10,ndcg10])

if __name__ == "__main__":
    main()