# Loan Approval Prediction System
# Utilizing Logistic Regression and Streamlit for an Interactive UI

# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("loan-train.csv")

    # Handle missing values
    data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)
    data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
    data["Credit_History"].fillna(data["Credit_History"].mode()[0], inplace=True)

    # Convert Loan_Status: "Y" → 1, "N" → 0
    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

    # Convert categorical features using One-Hot Encoding
    data = pd.get_dummies(data, columns=["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"], drop_first=True)

    return data

# Load data
data = load_data()

# Define features & target variable
features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
X = data[features]
y = data["Loan_Status"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict with scaled test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit App Interface ---
st.title("Loan Approval Prediction System")
st.write("Predict if a loan will be approved using machine learning.")

# Display Dataset Preview
st.subheader("Dataset Preview")
st.write(data.head())

# Show Model Accuracy
st.subheader("Model Performance")
st.write(f"Model Accuracy: {accuracy:.2%}")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Denied", "Approved"], yticklabels=["Denied", "Approved"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
feature_importance = np.abs(model.coef_[0])
features_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
features_df = features_df.sort_values(by="Importance", ascending=False)
st.write(features_df)

# Visualization: Loan Amount, Credit History & Loan Approval vs. Income
st.subheader("Data Visualization")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loan Amount Distribution
sns.histplot(data["LoanAmount"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Distribution of Loan Amounts")

# Credit History Countplot
sns.countplot(x="Credit_History", data=data, ax=axes[1])
axes[1].set_title("Loan Applications by Credit History")

# Loan Approval vs. Income
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=data, ax=axes[2])
axes[2].set_title("Loan Approval vs. Applicant Income")
axes[2].set_xticklabels(["Denied", "Approved"])

st.pyplot(fig)


# --- User Input for Loan Prediction ---
st.subheader("Enter Loan Applicant Details")
applicant_income = st.number_input("Applicant Income ($)", min_value=0, step=500)
coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, step=500)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, step=100)
loan_term = st.selectbox("Loan Term (Months)", [360, 180, 120, 60])
credit_history = st.radio("Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Bad")

# Make Prediction
if st.button("Predict Loan Approval"):
    applicant_data = pd.DataFrame({
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history]
    })

    # Scale the user input before making predictions
    applicant_data_scaled = scaler.transform(applicant_data)

    prediction = model.predict(applicant_data_scaled)
    prediction_prob = model.predict_proba(applicant_data_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f"Loan Approved! Probability: {prediction_prob:.2%}")
    else:
        st.error(f"Loan Denied. Probability: {prediction_prob:.2%}")

# --- Instructions for Evaluators ---
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Ensure 'loan-train.csv' is in the project directory.
    2. Run the application using the command:
       `streamlit run app.py`
    3. Input applicant details and click 'Predict Loan Approval'.

    **Important Notes:**
    - This model is best suited for predicting **smaller loan amounts** and **lower-income applicants**.
    - Larger loans and very high incomes may cause inaccurate predictions due to the nature of Logistic Regression.
    """
)

# Save requirements.txt for deployment
with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nscikit-learn\nnumpy\nmatplotlib\nseaborn")
