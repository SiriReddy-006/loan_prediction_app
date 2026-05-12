import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load datasets

df1 = pd.read_csv("loan_prediction.csv")
df2 = pd.read_csv("loan_status.csv")

# Combine datasets

df1['Loan_Status'] = df2['Loan_Status']

df = df1

# Fill null values

df.fillna(df.mode().iloc[0], inplace=True)

# Label Encoding

encoder = LabelEncoder()

columns = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for col in columns:
    df[col] = encoder.fit_transform(df[col])

# Separate Independent and Dependent Variables

x = df.drop(['Loan_ID', 'Loan_Status'], axis=1)

y = df['Loan_Status']

# Convert Dependents Column

x['Dependents'] = x['Dependents'].replace('3+', '3')

x['Dependents'] = pd.to_numeric(x['Dependents'])

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=100,
    stratify=y
)

# Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=100
)

# Train Model

model.fit(x_train, y_train)

# Streamlit UI

st.title("Loan Prediction System")

with st.form("loan_form"):

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

    applicant_income = st.number_input(
        "Applicant Income (₹)",
        min_value=0,
        step=1000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (₹)",
        min_value=0,
        step=1000
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        step=1000
    )

    loan_term = st.number_input(
        "Loan Amount Term (Months)",
        min_value=0,
        step=12
    )

    credit_history = st.selectbox(
        "Credit History",
        [0, 1]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    submit = st.form_submit_button("Predict Loan Status")

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

    input_data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")

    else:
        st.error("❌ Loan Rejected")