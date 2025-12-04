
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import load_npz

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data"

EXCEL_PATH = DATA_PATH / "online_course_recommendation_v2.xlsx"

@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH / "mappings.pkl", "rb") as f:
        maps = pickle.load(f)
    user2idx = maps["user2idx"]
    idx2user = maps["idx2user"]
    item2idx = maps["item2idx"]
    idx2item = maps["idx2item"]

    user_item = load_npz(ARTIFACTS_PATH / "user_item.npz")
    return user2idx, idx2user, item2idx, idx2item, user_item


@st.cache_resource
def load_courses():
    if EXCEL_PATH.exists():
        return pd.read_excel(EXCEL_PATH)
    return None

def recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df=None):
    user_id = str(user_id)
    if user_id not in user2idx:
        return pd.DataFrame(columns=["course_id", "score"])

    uidx = user2idx[user_id]
    user_row = user_item[uidx]
    interacted = set(user_row.indices)

    popularity = np.array(user_item.sum(axis=0)).ravel()
    popularity[list(interacted)] = -np.inf

    top_idx = np.argsort(popularity)[::-1][:top_k]
    top_items = [idx2item[i] for i in top_idx]
    top_scores = popularity[top_idx]

    df = pd.DataFrame({"course_id": top_items, "score": top_scores})
    if courses_df is not None and "course_id" in courses_df.columns:
        df = df.merge(courses_df, on="course_id", how="left")
    return df

def main():
    st.title("📘 Online Course Recommendation System")

    user2idx, idx2user, item2idx, idx2item, user_item = load_artifacts()
    courses_df = load_courses()

    st.sidebar.header("Settings")
    sample_users = list(user2idx.keys())

    mode = st.sidebar.radio("Input mode", ["Select user", "Type user"])

    if mode == "Select user" and sample_users:
        user_id = st.sidebar.selectbox("User ID", sample_users)
    else:
        user_id = st.sidebar.text_input("User ID")

    top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10)

    if st.button("Recommend"):
        recs = recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df)
        if recs.empty:
            st.warning("No results (invalid user or no data).")
        else:
            st.dataframe(recs)

if __name__ == "__main__":
    main()
