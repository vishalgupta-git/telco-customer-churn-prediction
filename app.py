import streamlit as st
import pandas as pd
import joblib


with open("./bin/label_encoders.pkl", "rb") as f:
    label_encoders = joblib.load(f)

with open("./bin/gradient_boosting_model.pkl", "rb") as f:
    model = joblib.load(f)


st.set_page_config(page_title="Customer Churn Predictor", layout="centered", page_icon="📞")

st.image(
    "https://images.unsplash.com/photo-1525182008055-f88b95ff7980",
    use_container_width=True
)
st.title("📞 Interactive Telecom Customer Churn Prediction")



st.markdown("---")

with st.form("churn_form"):
    st.header("👤 Customer Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.radio("Gender", ['Female', 'Male'], horizontal=True)
    with c2:
        seniorcitizen = "Yes" if st.toggle("Senior Citizen", value=False) else "No"
    with c3:
        tenure = st.slider("Tenure (months)", 0, 72, 24, step=1)

    st.header("🏠 Family & Dependents")
    c1, c2 = st.columns(2)
    with c1:
        partner = st.radio("Has Partner?", ['No', 'Yes'], horizontal=True)
    with c2:
        dependents = st.radio("Has Dependents?", ['No', 'Yes'], horizontal=True)

    st.header("📞 Services & Internet")
    with st.expander("Phone & Internet Services"):
        c1, c2 = st.columns(2)
        with c1:
            phoneservice = st.selectbox("Phone Service", ['No', 'Yes'])
            multiplelines = st.select_slider("Multiple Lines", ['No phone service', 'No', 'Yes'])
            internetservice = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        with c2:
            onlinesecurity = st.radio("Online Security", ['No', 'No internet service', 'Yes'])
            onlinebackup = st.radio("Online Backup", ['No', 'No internet service', 'Yes'])
            deviceprotection = st.radio("Device Protection", ['No', 'No internet service', 'Yes'])
            techsupport = st.radio("Tech Support", ['No', 'No internet service', 'Yes'])
            streamingtv = st.radio("Streaming TV", ['No', 'No internet service', 'Yes'])
            streamingmovies = st.radio("Streaming Movies", ['No', 'No internet service', 'Yes'])

    st.header("💳 Contract & Billing")
    with st.expander("Billing Information"):
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = "Yes" if st.toggle("Paperless Billing", value=True) else "No"
        paymentmethod = st.radio(
            "Payment Method",
            [
                'Bank transfer (automatic)',
                'Credit card (automatic)',
                'Electronic check',
                'Mailed check'
            ],
            horizontal=False,
        )
        c1, c2 = st.columns(2)
        with c1:
            monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1500.0, value=70.0, step=1.0)
        with c2:
            totalcharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=2500.0, step=10.0)

    submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

# -------------------------
# Prediction
# -------------------------
if submitted:
    input_data = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
    }

    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Prediction
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error("🔴 **Customer likely to churn**")
        else:
            st.success("🟢 **Customer likely to stay**")
    # with col2:
    #     if proba is not None:
    #         st.metric("Churn Probability", f"{proba * 100:.2f}%")

    with st.expander("📋 Encoded Input Data"):
        st.dataframe(df.T)

    st.balloons()
