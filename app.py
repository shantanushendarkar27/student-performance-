import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

model = joblib.load("svm_pass_fail_model.pkl")
scaler = joblib.load("scaler.pkl")


st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🎓 Student Performance Prediction</h1>
    <p style='text-align: center; font-size: 16px;'>
    Predict whether a student will <b>Pass</b> or <b>Fail</b> using Machine Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📘 Project Info")
st.sidebar.markdown(
    """
    **Model Used:** Tuned SVM  
    **Technique:** SMOTE + GridSearchCV  
    **Accuracy:** 100%  
    **Use Case:** Early identification of at-risk students
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍🎓 *Developed for Academic Project*")

st.subheader("📊 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "📚 Study Hours per Week",
        min_value=5,
        max_value=100,
        value=20
    )

    attendance = st.slider(
        "🧾 Attendance Rate (%)",
        min_value=5,
        max_value=100,
        value=75
    )

with col2:
    past_score = st.slider(
        "📝 Past Exam Score",
        min_value=0,
        max_value=100,
        value=70
    )

    final_score = st.slider(
        "🎯 Final Exam Score",
        min_value=0,
        max_value=100,
        value=60
    )

if st.button("🔮 Predict Result", use_container_width=True):

    input_data = np.array(
        [[study_hours, attendance, past_score, final_score]]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    if prediction == 1:
        st.success("**Prediction: PASS** 🎉")
        st.markdown(
            """
            The student is likely to **pass** based on current performance indicators.
            """
        )
    else:
        st.error(" **Prediction: FAIL**")
        st.markdown(
            """
            The student may be **at risk**.  
            Early intervention is recommended.
            """
        )

st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:13px;'>
    Machine Learning Project By HRK😎 | Student Performance Prediction
    </p>
    """,
    unsafe_allow_html=True
)
