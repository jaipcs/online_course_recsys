import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from scipy import sparse
import plotly.express as px   # for radar & bar charts

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Course Recommendation System", layout="wide")

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data"
EXCEL_PATH = DATA_PATH / "online_course_recommendation_v2.xlsx"

# -------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------
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
    """Load Excel and normalize column names to have course_id and course_name."""
    if not EXCEL_PATH.exists():
        return None

    df = pd.read_excel(EXCEL_PATH)

    # Normalize column names to lower + strip
    col_map = {c.lower().strip(): c for c in df.columns}

    # ----- Ensure 'course_id' exists -----
    if "course_id" not in df.columns:
        for cand in ["course id", "courseid", "course_id", "id", "course"]:
            if cand in col_map:
                df = df.rename(columns={col_map[cand]: "course_id"})
                break

    # ----- Ensure 'course_name' exists -----
    if "course_name" not in df.columns:
        for cand in [
            "course name",
            "coursename",
            "course_title",
            "course title",
            "title",
            "name",
        ]:
            if cand in col_map:
                df = df.rename(columns={col_map[cand]: "course_name"})
                break

    # Fallbacks
    if "course_id" in df.columns and "course_name" not in df.columns:
        df["course_name"] = df["course_id"].astype(str)

    if "course_name" in df.columns:
        df["course_name"] = df["course_name"].fillna(
            df.get("course_id", "").astype(str)
        )

    return df

# -------------------------------------------------------------------
# BASELINE RECOMMENDER (popularity-based)
# -------------------------------------------------------------------
def recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df=None):
    user_id = str(user_id)
    if user_id not in user2idx:
        return pd.DataFrame(columns=["course_id", "score", "course_name"])

    uidx = user2idx[user_id]

    # user history row
    user_row = user_item[uidx]
    interacted = set(user_row.indices)

    # popularity-based baseline
    popularity = np.array(user_item.sum(axis=0)).ravel()
    # do not recommend already interacted items
    popularity[list(interacted)] = -np.inf

    top_idx = np.argsort(popularity)[::-1][:top_k]
    top_items = [idx2item[i] for i in top_idx]
    top_scores = popularity[top_idx]

    df = pd.DataFrame({"course_id": top_items, "score": top_scores})

    if courses_df is not None and "course_id" in courses_df.columns:
        df = df.merge(courses_df, on="course_id", how="left")

    # Ensure course_name is always present
    if "course_name" not in df.columns:
        df["course_name"] = df["course_id"].astype(str)
    else:
        df["course_name"] = df["course_name"].fillna(df["course_id"].astype(str))

    return df

# -------------------------------------------------------------------
# MODEL METRICS (FILL WITH YOUR REAL VALUES)
# -------------------------------------------------------------------
METRICS = {
    "Content-based": {
        "precision": 0.32,
        "recall": 0.21,
        "f1": 0.25,
        "rmse": 0.95,
        "mae": 0.74,
    },
    "Item-CF": {
        "precision": 0.35,
        "recall": 0.24,
        "f1": 0.28,
        "rmse": 0.92,
        "mae": 0.70,
    },
    "Hybrid": {
        "precision": 0.40,
        "recall": 0.28,
        "f1": 0.33,
        "rmse": 0.88,
        "mae": 0.66,
    },
}

# -------------------------------------------------------------------
# VISUAL HELPERS
# -------------------------------------------------------------------
def plot_metrics_radar(metrics_dict):
    """
    metrics_dict: {
       'ModelName': {'precision':..., 'recall':..., 'f1':..., 'rmse':..., 'mae':...},
       ...
    }
    """
    rows = []
    for model_name, m in metrics_dict.items():
        for metric_name, value in m.items():
            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric_name.upper(),
                    "Score": float(value),
                }
            )

    df = pd.DataFrame(rows)

    fig = px.line_polar(
        df,
        r="Score",
        theta="Metric",
        color="Model",
        line_close=True,
        markers=True,
    )
    fig.update_traces(fill="toself", opacity=0.6)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Model Metrics Radar Plot",
    )
    return fig


def plot_reco_status_bar(recs_df):
    """
    recs_df: DataFrame with columns:
       ['course_id', 'course_name', 'score', 'status']
       where status ∈ {'New Recommendation', 'Already Taken'}
    """
    status_counts = recs_df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.bar(
        status_counts,
        x="Status",
        y="Count",
        text="Count",
    )
    fig.update_layout(
        title="Recommended vs Already Taken Courses",
        xaxis_title="Status",
        yaxis_title="Number of Courses",
    )
    fig.update_traces(textposition="outside")
    return fig


# -------------------------------------------------------------------
# HIGHER LEVEL RECOMMENDER FOR UI
# -------------------------------------------------------------------
def recommend_courses_for_user(user_id, top_k, user2idx, item2idx, idx2item, user_item, courses_df):
    """
    Wraps simple recommender and adds:
      - course_name
      - already_taken flag
    Returns list of dicts for the UI.
    """
    user_id = str(user_id)

    # if invalid user -> empty
    if user_id not in user2idx:
        return []

    uidx = user2idx[user_id]
    user_row = user_item[uidx]
    interacted_idx = set(user_row.indices)

    # get baseline recs
    recs_df = recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df)

    if recs_df.empty:
        return []

    recs = []
    for _, row in recs_df.iterrows():
        course_id = row["course_id"]

        # map course_id back to internal index (if possible)
        internal_idx = None
        if item2idx is not None:
            internal_idx = item2idx.get(str(course_id))

        already_taken = False
        if internal_idx is not None:
            already_taken = internal_idx in interacted_idx

        # strong fallback for name
        name = str(row.get("course_name", course_id))
        if pd.isna(name) or name.strip() == "":
            name = str(course_id)

        recs.append(
            {
                "course_id": course_id,
                "course_name": name,
                "score": float(row["score"]),
                "already_taken": already_taken,
            }
        )
    return recs

# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------
def main():
    st.title("🎓 Online Course Recommendation Chat")

    # Load data / artifacts
    user2idx, idx2user, item2idx, idx2item, user_item = load_artifacts()
    courses_df = load_courses()

    
    # ---------------- SIDEBAR ----------------
    st.sidebar.header("⚙️ Settings")

    sample_users = list(user2idx.keys())

    mode = st.sidebar.radio("Input mode", ["Select user", "Type user"])

    if mode == "Select user" and sample_users:
        user_id = st.sidebar.selectbox("User ID", sample_users)
    else:
        user_id = st.sidebar.text_input("User ID")

    top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Evaluation Metrics")

    # Metrics table
    metrics_df = pd.DataFrame(METRICS).T[
        ["precision", "recall", "f1", "rmse", "mae"]
    ]
    metrics_df = metrics_df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "rmse": "RMSE",
            "mae": "MAE",
        }
    )
    st.sidebar.subheader("Metrics Table")
    st.sidebar.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    # Radar plot
    radar_fig = plot_metrics_radar(METRICS)
    st.sidebar.subheader("Radar Plot")
    st.sidebar.plotly_chart(radar_fig, use_container_width=True)

    
    # ---------------- MAIN LAYOUT ----------------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("💬 Recommendation Chat")

        if st.button("Recommend Courses"):
            recs = recommend_courses_for_user(
                user_id=user_id,
                top_k=top_k,
                user2idx=user2idx,
                item2idx=item2idx,
                idx2item=idx2item,
                user_item=user_item,
                courses_df=courses_df,
            )

            if not recs:
                st.warning("No recommendations found for this user.")
            else:
                recs_df = pd.DataFrame(recs)

                # Add human-readable status
                if "already_taken" in recs_df.columns:
                    recs_df["status"] = recs_df["already_taken"].apply(
                        lambda x: "Already Taken" if x else "New Recommendation"
                    )
                else:
                    recs_df["status"] = "New Recommendation"

                st.markdown("### ✅ Recommended Courses")
                st.dataframe(
                    recs_df[["course_name", "score", "status"]]
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                    .style.format({"score": "{:.4f}"}),
                    use_container_width=True,
                )

                # save for right-column visuals
                st.session_state["last_recs_df"] = recs_df

    with col_right:
        st.subheader("📈 Recommendation Visuals")

        recs_df = st.session_state.get("last_recs_df", None)
        if recs_df is not None:
            # 1) Bar chart: new vs already taken
            bar_fig = plot_reco_status_bar(recs_df)
            st.plotly_chart(bar_fig, use_container_width=True)

            # 2) Score distribution per course
            st.markdown("### Score Distribution")
            score_fig = px.bar(
                recs_df.sort_values("score", ascending=False),
                x="course_name",
                y="score",
                color="status",
            )
            score_fig.update_layout(
                xaxis_title="Course",
                yaxis_title="Recommendation Score",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(score_fig, use_container_width=True)
        else:
            st.info("Run a recommendation to see visualizations.")


if __name__ == "__main__":
    main()
