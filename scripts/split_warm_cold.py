import argparse, numpy as np, pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
OUT_ROOT = Path("splits")

def strict_cold_split(ratings, cold_user_ratio=0.1, cold_item_ratio=0.0, valid_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    users = ratings["user_id"].unique()
    items = ratings["item_id"].unique()

    n_cu = int(len(users) * cold_user_ratio)
    n_ci = int(len(items) * cold_item_ratio)
    cold_users = set(rng.choice(users, n_cu, replace=False)) if n_cu > 0 else set()
    cold_items = set(rng.choice(items, n_ci, replace=False)) if n_ci > 0 else set()

    # ✅ 핵심: cold row 마스크로 분리
    is_cold = ratings["user_id"].isin(cold_users) | ratings["item_id"].isin(cold_items)

    test_cold = ratings[is_cold].copy().reset_index(drop=True)
    remain = ratings[~is_cold].copy().reset_index(drop=True)

    # valid는 remain에서 샘플링
    valid = remain.sample(frac=valid_ratio, random_state=seed)
    train = remain.drop(valid.index).reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    print(f"[seed={seed}] ratings={len(ratings)}  train={len(train)}  valid={len(valid)}  test_cold={len(test_cold)}")
    return train, valid, test_cold, cold_users, cold_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold_user_ratio", type=float, default=0.1)
    ap.add_argument("--cold_item_ratio", type=float, default=0.0)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    ratings = pd.read_csv(DATA_DIR/"ratings.csv")
    train, valid, test_cold, cold_users, cold_items = strict_cold_split(
        ratings, args.cold_user_ratio, args.cold_item_ratio, args.valid_ratio, args.seed
    )

    outdir = Path(args.outdir) if args.outdir else OUT_ROOT / f"seed{args.seed}_u{args.cold_user_ratio}_i{args.cold_item_ratio}"
    outdir.mkdir(parents=True, exist_ok=True)

    train.to_csv(outdir/"train.csv", index=False)
    valid.to_csv(outdir/"valid.csv", index=False)
    test_cold.to_csv(outdir/"test_cold.csv", index=False)
    pd.Series(sorted(list(cold_users)), name="cold_user_id").to_csv(outdir/"cold_users.csv", index=False)
    pd.Series(sorted(list(cold_items)), name="cold_item_id").to_csv(outdir/"cold_items.csv", index=False)

    print(f"[+] Saved to {outdir}")

if __name__ == "__main__":
    main()