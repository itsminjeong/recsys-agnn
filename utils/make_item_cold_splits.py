# utils/make_item_cold_splits.py
# MovieLens-100K ratings.csv (user_id,item_id,rating,timestamp) 기준
# seed, u_ratio, i_ratio 조합으로 splits/seed{S}_u{u}_i{i}/train|valid|test_cold.csv 생성

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ratings", default="data/processed/ratings.csv")
ap.add_argument("--seeds", nargs="+", type=int, default=[42])
ap.add_argument("--user_ratios", nargs="+", type=float, default=[0.0])
ap.add_argument("--item_ratios", nargs="+", type=float, default=[0.1, 0.3, 0.5])
args = ap.parse_args()

ratings = pd.read_csv(args.ratings)
ratings = ratings[["user_id", "item_id", "rating"]]

def make_split(seed, u_ratio, i_ratio):
    np.random.seed(seed)
    users = ratings.user_id.unique()
    items = ratings.item_id.unique()

    cold_users = set(np.random.choice(users, int(len(users) * u_ratio), replace=False)) if u_ratio > 0 else set()
    cold_items = set(np.random.choice(items, int(len(items) * i_ratio), replace=False)) if i_ratio > 0 else set()

    # train: cold user/item 포함하지 않음
    mask_cold_ui = ratings.user_id.isin(cold_users) | ratings.item_id.isin(cold_items)
    train = ratings[~mask_cold_ui].copy()

    # cold eval set: cold user OR cold item 포함
    cold = ratings[mask_cold_ui].copy()

    # valid: train 중 10% 샘플
    tr_idx = np.arange(len(train))
    np.random.shuffle(tr_idx)
    n_valid = max(1, int(0.1 * len(tr_idx)))
    valid = train.iloc[tr_idx[:n_valid]].copy()
    train = train.iloc[tr_idx[n_valid:]].copy()

    split_dir = Path(f"splits/seed{seed}_u{u_ratio}_i{i_ratio}")
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.csv", index=False)
    valid.to_csv(split_dir / "valid.csv", index=False)
    cold.to_csv(split_dir / "test_cold.csv", index=False)
    print(f"✅ made: {split_dir} | rows: train {len(train)}, valid {len(valid)}, test_cold {len(cold)}")

for s in args.seeds:
    # user-cold only
    for u in args.user_ratios:
        make_split(s, u_ratio=u, i_ratio=0.0)
    # item-cold only
    for i in args.item_ratios:
        make_split(s, u_ratio=0.0, i_ratio=i)