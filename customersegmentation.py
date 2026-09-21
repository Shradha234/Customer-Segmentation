import streamlit as st
import joblib
import numpy as np

# Load KMeans model
model = joblib.load("kmeans_model.pkl")

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# Title
st.title("Customer Segmentation App")

st.write("Enter customer details to find which segment they belong to")

# ---- INPUT ----
income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    max_value=5000.0,
    value=50.0,
    step=1.0
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1.0,
    max_value=900.0,
    value=50.0,
    step=1.0
)

# ---- Prediction ----
if st.button("Predict Cluster"):
    try:
        input_data = np.array([[income, score]])
        cluster = model.predict(input_data)

        cluster_id = int(cluster[0])
   # Optional: Meaning of clusters
        cluster_meaning = {
            0: "Low Income, Low Spending",
            1: "High Income, High Spending",
            2: "Average Customers",
            3: "High Income, Low Spending",
            4: "Low Income, High Spending"
        }

        st.success(f"Customer belongs to Cluster {cluster_id}")
        st.info(f"Segment: {cluster_meaning.get(cluster_id, 'Unknown')}")

    except Exception as e:
        st.error(f"Error: {e}")