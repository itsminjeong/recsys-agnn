"""
make_split.py
- ratings.csv에서 cold-user/valid 비율에 맞춰 split 폴더 생성
- 생성 파일: train.csv, valid.csv, test_cold.csv, cold_users.csv, cold_items.csv
- 기존 10% split과 동일한 스키마(user_id,item_id,rating[,timestamp]) 유지
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/processed/ratings.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cold_user_ratio", type=float, default=0.1)
    ap.add_argument("--cold_item_ratio", type=float, default=0.0)  # 필요 없으면 0.0
    ap.add_argument("--valid_ratio", type=float, default=0.1)      # warm 유저의 valid 비율
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드 (필수 컬럼: user_id, item_id, rating / timestamp는 있으면 유지)
    ratings = pd.read_csv(args.ratings)
    needed = ["user_id", "item_id", "rating"]
    for c in needed:
        if c not in ratings.columns:
            raise ValueError(f"ratings.csv에 '{c}' 컬럼이 필요합니다.")
    extra_cols = [c for c in ["timestamp"] if c in ratings.columns]
    cols = needed + extra_cols
    ratings = ratings[cols].copy()

    # 2) cold users 선택
    users = ratings["user_id"].unique()
    n_cold_u = int(len(users) * args.cold_user_ratio)
    cold_users = set(rng.choice(users, size=n_cold_u, replace=False)) if n_cold_u > 0 else set()

    # 3) (옵션) cold items 선택
    items = ratings["item_id"].unique()
    n_cold_i = int(len(items) * args.cold_item_ratio)
    cold_items = set(rng.choice(items, size=n_cold_i, replace=False)) if n_cold_i > 0 else set()

    # 4) split
    # cold test: cold_users 또는 cold_items에 해당하는 상호작용 모두 이동
    is_cold_row = ratings["user_id"].isin(cold_users) | ratings["item_id"].isin(cold_items)
    test_cold = ratings[is_cold_row].copy()
    warm_pool = ratings[~is_cold_row].copy()

    # warm 유저 기준 valid 분리 (유저별로 나눠서 시간정보 없으면 무작위로)
    warm_users = warm_pool["user_id"].unique()
    valid_idx = []
    for u in warm_users:
        u_rows = warm_pool[warm_pool["user_id"] == u]
        idx = u_rows.index.to_numpy()
        rng.shuffle(idx)
        k = int(len(idx) * args.valid_ratio)
        if k > 0:
            valid_idx.extend(idx[:k])
    valid = warm_pool.loc[valid_idx].copy()
    train = warm_pool.drop(index=valid_idx).copy()

    # 5) 저장
    train.to_csv(out_dir / "train.csv", index=False)
    valid.to_csv(out_dir / "valid.csv", index=False)
    test_cold.to_csv(out_dir / "test_cold.csv", index=False)

    # 부가 메타(한 줄짜리 리스트 형태)
    pd.DataFrame(sorted(list(cold_users)), columns=["user_id"]).to_csv(out_dir / "cold_users.csv", index=False)
    pd.DataFrame(sorted(list(cold_items)), columns=["item_id"]).to_csv(out_dir / "cold_items.csv", index=False)

    # 로그
    print(f"[OK] Split saved to: {out_dir}")
    print(f" - users: total {len(users)}, cold {len(cold_users)} ({args.cold_user_ratio:.0%})")
    print(f" - items: total {len(items)}, cold {len(cold_items)} ({args.cold_item_ratio:.0%})")
    print(f" - rows: train {len(train)}, valid {len(valid)}, test_cold {len(test_cold)}")

if __name__ == "__main__":
    main()