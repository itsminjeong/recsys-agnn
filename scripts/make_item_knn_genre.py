import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_knn_topk(X: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (I, G) float32
    return:
      nbrs: (I, K) int64  - 각 i의 topK 이웃 item index
      sims: (I, K) float32 - 각 i와 이웃의 cosine similarity
    """
    # L2 normalize (cosine = dot after normalize)
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norm

    I = Xn.shape[0]
    nbrs = np.zeros((I, K), dtype=np.int64)
    sims = np.zeros((I, K), dtype=np.float32)

    # 아이템 수가 크면 블록으로 하는 게 안전하지만,
    # MovieLens-100K(I~1682)는 전체 dot도 충분히 가능.
    S = Xn @ Xn.T  # (I, I)
    np.fill_diagonal(S, -1.0)  # self-edge 제거

    # 각 행에서 topK 뽑기 (argpartition -> 정렬)
    for i in range(I):
        row = S[i]
        if K >= I:
            idx = np.argsort(-row)
            idx = idx[: I - 1]
        else:
            cand = np.argpartition(-row, K)[:K]  # topK 후보
            idx = cand[np.argsort(-row[cand])]   # 후보 내 정렬
        nbrs[i] = idx
        sims[i] = row[idx].astype(np.float32)

    return nbrs, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_csv", default="data/processed/items.csv")
    ap.add_argument("--out_csv", default="data/processed/item_item_knn.csv")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min_sim", type=float, default=0.0,
                    help="cosine sim이 이 값보다 작은 edge는 버림(처음엔 0.0 추천)")
    args = ap.parse_args()

    items_path = Path(args.items_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = pd.read_csv(items_path)

    # item_id가 0..I-1로 연속이라는 가정(너 split도 이 기준으로 돌아가고 있음)
    if "item_id" not in items.columns:
        raise RuntimeError("items.csv에 item_id 컬럼이 없습니다.")

    genre_cols = [c for c in items.columns if c.startswith("genre_")]
    if not genre_cols:
        raise RuntimeError("items.csv에서 genre_* 컬럼을 찾지 못했습니다.")

    # item_id 정렬 보장
    items = items.sort_values("item_id").reset_index(drop=True)

    X = items[genre_cols].to_numpy(dtype=np.float32)  # (I, G)
    I = X.shape[0]
    if args.k >= I:
        raise ValueError(f"K={args.k}는 아이템 수 I={I}보다 작아야 합니다.")

    nbrs, sims = cosine_knn_topk(X, args.k)

    # long-form edge list로 변환: (item_i, item_j, sim)
    rows = []
    for i in range(I):
        for j_idx, s in zip(nbrs[i], sims[i]):
            if s >= args.min_sim:
                rows.append((i, int(j_idx), float(s)))

    df_out = pd.DataFrame(rows, columns=["item_i", "item_j", "sim"])
    df_out.to_csv(out_path, index=False)

    print(f"[OK] saved: {out_path}  (edges={len(df_out):,}, K={args.k}, min_sim={args.min_sim})")
    print(df_out.head())


if __name__ == "__main__":
    main()