# run_mf.py
# 목적:
# - MF (속성 OFF) 학습
# - Warm(valid): RMSE, MAE, NDCG@10
# - Cold(test_cold): RMSE, MAE, Recall@10, NDCG@10
# - 결과를 results/metrics.csv에 누적 저장

import argparse, os, csv
from pathlib import Path
import pandas as pd
import numpy as np
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

    def __len__(self): return self.r.shape[0]
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

# -----------------------------
# Model (MF)
# -----------------------------
class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

    def forward(self, u, i):
        # u,i: LongTensor
        logits = (self.U(u) * self.V(i)).sum(-1)
        logits = logits + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.mu
        return logits

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/seed42_u0.1_i0.0")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4096)
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

    # Dataloaders
    tr_loader = DataLoader(RatingsDS(tr), batch_size=args.bs, shuffle=True, num_workers=0)
    va_loader = DataLoader(RatingsDS(va), batch_size=args.bs, shuffle=False, num_workers=0)
    te_loader = DataLoader(RatingsDS(te), batch_size=args.bs, shuffle=False, num_workers=0)

    # Model
    model = MF(n_users, n_items, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best = (1e9, 1e9, -1.0)  # (RMSE, MAE, NDCG@10)
    best_ep = 0

    # Train
    for ep in range(1, args.epochs + 1):
        model.train()
        for u, i, r in tr_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            opt.zero_grad(); loss.backward(); opt.step()

        # Warm eval
        vrmse, vmae = eval_rmse_mae(model, va_loader, device)
        vndcg = eval_warm_ndcg_at_k(model, tr, va, n_items, K=10, device=device)
        print(f"[Ep {ep:02d}] valid RMSE={vrmse:.4f} MAE={vmae:.4f} NDCG@10={vndcg:.4f}")

        if vrmse < best[0]:
            best = (vrmse, vmae, vndcg)
            best_ep = ep

    # Cold eval
    trmse, tmae = eval_rmse_mae(model, te_loader, device)
    recall10, ndcg10 = eval_cold_recall_ndcg_at_k(model, tr, te, n_items, K=10, device=device)
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
        w.writerow(["MF","OFF",str(split_dir),args.dim,best_ep,
                    best[0],best[1],best[2],
                    trmse,tmae,recall10,ndcg10])

if __name__ == "__main__":
    main()