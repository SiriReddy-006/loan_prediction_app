import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Dataset

df = pd.read_csv("loan_prediction.csv")

# Fill Null Values

df.fillna(df.mode().iloc[0], inplace=True)

# Label Encoding

encoder = LabelEncoder()

columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area'
]

for col in columns:
    df[col] = encoder.fit_transform(df[col])

# Convert Dependents Column

df['Dependents'] = df['Dependents'].replace('3+', '3')

df['Dependents'] = pd.to_numeric(df['Dependents'])

# Features and Target

x = df.drop(['Loan_ID', 'LoanAmount'], axis=1)

y = df['LoanAmount']

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Regressor Model

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train Model

model.fit(x_train, y_train)

# Streamlit UI

st.title("Loan Amount Prediction using Random Forest Regressor")

with st.form("prediction_form"):

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        [0, 1, 2, 3]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.text_input(
        "Applicant Income",
        "25000"
    )

    st.caption("Enter integer values only")

    coapplicant_income = st.text_input(
        "Coapplicant Income",
        "10000"
    )

    st.caption("Enter integer values only")

    loan_term = st.text_input(
        "Loan Amount Term (Months)",
        "360"
    )

    st.caption("Enter integer values only")

    credit_history = st.selectbox(
        "Credit History",
        [0, 1]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    submit = st.form_submit_button("Predict Loan Amount")

# Manual Encoding

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Rural":
    property_area = 0

elif property_area == "Semiurban":
    property_area = 1

else:
    property_area = 2

# Prediction

if submit:

    # Convert Text Inputs to Integer

    applicant_income = int(applicant_income)

    coapplicant_income = int(coapplicant_income)

    loan_term = int(loan_term)

    input_data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_term,
        credit_history,
        property_area
    ]]

    prediction = model.predict(input_data)

    # Convert Thousands to Actual Amount

    predicted_amount = int(prediction[0] * 1000)

    st.success(
        f"Estimated Loan Amount: ₹{predicted_amount:,}"
    )