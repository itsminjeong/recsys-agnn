import pandas as pd
import os

RAW_PATH = "data/ml-100k/ml-100k" 
OUT_PATH = "data/processed" 

os.makedirs(OUT_PATH, exist_ok=True)

def load_data(): 
    ratings = pd.read_csv(
        os.path.join(RAW_PATH, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        os.path.join(RAW_PATH, "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"]
    )
    items = pd.read_csv(
        os.path.join(RAW_PATH, "u.item"),
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] +
              [f"genre_{i}" for i in range(19)]
    )
    return ratings, users, items

def preprocess():
    ratings, users, items = load_data()
    print("원본 데이터 크기:", len(ratings), "ratings,", len(users), "users,", len(items), "items")

    # 사용자/아이템 id를 0부터 시작하도록 매핑
    user_id_map = {id: idx for idx, id in enumerate(users["user_id"].unique())}
    item_id_map = {id: idx for idx, id in enumerate(items["item_id"].unique())}

    ratings["user_id"] = ratings["user_id"].map(user_id_map)
    ratings["item_id"] = ratings["item_id"].map(item_id_map)

    # 저장
    ratings.to_csv(os.path.join(OUT_PATH, "ratings.csv"), index=False)
    users.to_csv(os.path.join(OUT_PATH, "users.csv"), index=False)
    items.to_csv(os.path.join(OUT_PATH, "items.csv"), index=False)

    print("전처리 완료! → data/processed/ 에 저장됨")

if __name__ == "__main__":
    preprocess()