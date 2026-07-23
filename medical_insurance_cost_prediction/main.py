from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.figure_factory as ff
BASE_DIR = Path(__file__).parent

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Insurance Cost Suite",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Session State Initialization
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Load Assets & Preprocess
# -----------------------------
@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_and_process_data():
    raw_df = pd.read_csv(BASE_DIR / "insurance.csv")
    
    # Keep a copy with clean labels for plotting
    display_df = raw_df.copy()
    display_df["sex"] = display_df["sex"].str.capitalize()
    display_df["smoker"] = display_df["smoker"].map({"yes": "Smoker", "no": "Non-Smoker"})
    display_df["region"] = display_df["region"].str.capitalize()
    
    return display_df

try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not find `model.pkl`. Please ensure the file is present in your working directory.")
    st.stop()

df_display = load_and_process_data()
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["About Project", "Exploratory Data Analysis", "Insurance Prediction"]
)

# ======================================================
# Page 1: About Project
# ======================================================
if page == "About Project":
    st.title("🏥 Medical Insurance Charges Analysis Suite")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📌 Project Overview")
        st.write(
            "Healthcare billing models can be complex and obscure. This project leverages "
            "Machine Learning to transparently evaluate how key personal attributes—such as age, "
            "biometric markers, habits, and demographics—directly impact annualized medical insurance premiums."
        )
        
        st.subheader("🎯 Target Objectives")
        st.markdown(
            """
            * **Interpretability:** Quantify how physiological indicators (BMI) and lifestyle choices (smoking status) alter risk assessments.
            * **Deployability:** Provide an interface allowing users to dynamically forecast out-of-pocket pricing adjustments instantly.
            """
        )
        
    with col2:
        st.subheader("🛠️ Technology Stack")
        st.markdown(
            """
            - **Core:** Python, Pandas, NumPy
            - **Modeling:** Scikit-Learn (Linear Regression)
            - **Frontend Framework:** Streamlit
            - **Interactive Visualizations:** Plotly Express
            """
        )

# ======================================================
# Page 2: Exploratory Data Analysis
# ======================================================
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    st.subheader("Inspecting the Raw Records")
    st.dataframe(df_display, use_container_width=True)
    
    csv = df_display.to_csv(index=False).encode("utf-8")
   st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="insurance_clean.csv",
    mime="text/csv"
    )

    
    
    st.markdown("---")
    st.subheader("Categorical Split vs. Smoking Status")
    
    cat_col1, cat_col2, cat_col3 = st.columns(3)
    
    with cat_col1:
        fig_sex = px.histogram(df_display, x="sex", color="smoker", barmode="group", title="Gender Distribution")
        st.plotly_chart(fig_sex, use_container_width=True)
        
    with cat_col2:
        fig_child = px.histogram(df_display, x="children", color="smoker", barmode="group", title="Children Variance")
        st.plotly_chart(fig_child, use_container_width=True)
        
    with cat_col3:
        fig_region = px.histogram(df_display, x="region", color="smoker", barmode="group", title="Regional Spread")
        st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("---")
    st.subheader("Continuous Feature Distributions")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        fig_age_dist = ff.create_distplot([df_display["age"]], ["Age Group"], colors=['#1f77b4'])
        fig_age_dist.update_layout(title="Age Distribution Profile")
        st.plotly_chart(fig_age_dist, use_container_width=True)
        
    with dist_col2:
        fig_bmi_dist = ff.create_distplot([df_display["bmi"]], ["Body Mass Index"], colors=['#2ca02c'])
        fig_bmi_dist.update_layout(title="BMI Distribution Profile")
        st.plotly_chart(fig_bmi_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("Premium Correlations and Outliers")
    
    scat_col1, scat_col2 = st.columns(2)
    
    with scat_col1:
        fig_age_scat = px.scatter(df_display, x="age", y="charges", color="smoker", trendline="ols", title="Impact of Age on Premium Fees")
        st.plotly_chart(fig_age_scat, use_container_width=True)
        
    with scat_col2:
        fig_bmi_scat = px.scatter(df_display, x="bmi", y="charges", color="smoker", title="Impact of BMI on Premium Fees")
        st.plotly_chart(fig_bmi_scat, use_container_width=True)
        
    st.subheader("Financial Outlier Spread Across Demographics")
    fig_box = px.box(df_display, x="smoker", y="charges", color="sex", title="Charges Range by Smoking Status & Gender")
    st.plotly_chart(fig_box, use_container_width=True)

# ======================================================
# Page 3: Insurance Prediction
# ======================================================
else:
    st.title("🔮 Premium Estimator Engine")
    st.markdown("---")
    
    input_col, output_col = st.columns([1, 1])
    
    with input_col:
        st.subheader("User Attribute Settings")
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        sex = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of Dependent Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker Status", ["No", "Yes"])
        region = st.selectbox("Geographic Region", ["northeast", "northwest", "southeast", "southwest"])
        
        predict_btn = st.button("Calculate Premium Charges", type="primary")

    with output_col:
        st.subheader("Estimation Outputs")
        if predict_btn:
            # Prepare mathematical inputs mapping back to underlying system format
            numeric_sex = 1 if sex == "Male" else 0
            numeric_smoker = 1 if smoker == "Yes" else 0
            numeric_region = region_map[region]
            
            features = np.array([[age, numeric_sex, bmi, children, numeric_smoker, numeric_region]])
            prediction = model.predict(features)[0]
            
            st.success(f"### Estimated Value: **${prediction:,.2f}**")
            
            # Save into the session history stack
            st.session_state.history.append({
                "Age": age,
                "Gender": sex,
                "BMI": bmi,
                "Children": children,
                "Smoker": smoker,
                "Region": region.capitalize(),
                "Predicted Cost": f"${prediction:,.2f}"
            })
        else:
            st.info("Modify individual variables on the left pane and compute your forecast.")

    st.markdown("---")
    st.subheader("🕒 User Assessment History Log")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df.iloc[::-1], use_container_width=True) # View chronologically inverted (newest first)
        
        if st.button("Clear Cache History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No dynamic runs registered in this window session yet.")