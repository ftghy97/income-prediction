
import streamlit as st
import pickle
import pandas as pd

# Load model and column names
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

st.markdown("<h1 style='text-align:center;'>Income Category Prediction</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a person earns >50K or <=50K based on demographic and employment features.")

# Input form
def user_input():
    age = st.slider('Age', 17, 90, 30)
    hours_per_week = st.slider('Hours per Week', 1, 100, 40)
    capital_gain = st.number_input('Capital Gain', 0, 100000, 0)
    capital_loss = st.number_input('Capital Loss', 0, 5000, 0)
    fnlwgt = st.number_input('Final Weight', 10000, 1000000, 50000)

    workclass = st.selectbox('Workclass', [
        'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
        'Federal-gov', 'Self-emp-inc', 'Without-pay'
    ])

    education = st.selectbox('Education', [
        'Preschool', '1st-4th', '5th-6th', '7th-8th',
        '9th', '10th', '11th', '12th',
        'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm',
        'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
    ])

    education_num = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
        '9th': 5, '10th': 6, '11th': 7, '12th': 8,
        'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11,
        'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
        'Prof-school': 15, 'Doctorate': 16
    }[education]

    marital_status = st.selectbox('Marital Status', [
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Separated', 'Widowed', 'Married-spouse-absent'
    ])

    occupation = st.selectbox('Occupation', [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces'
    ])

    gender = st.selectbox('Gender', ['Male', 'Female'])
    relationship = st.selectbox('Relationship', [
        'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'
    ])
    race = st.selectbox('Race', [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])
    native_country = st.selectbox('Native Country', [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Other'
    ])

    data = {
        'Age': age,
        'Final Weight': fnlwgt,
        'EducationNum': education_num,
        'Capital Gain': capital_gain,
        'capital loss': capital_loss,
        'Hours per Week': hours_per_week,
        f'Workclass_{workclass}': 1,
        f'Education_{education}': 1,
        f'Marital Status_{marital_status}': 1,
        f'Occupation_{occupation}': 1,
        f'Gender_{gender}': 1,
        f'Relationship_{relationship}': 1,
        f'Race_{race}': 1,
        f'Native Country_{native_country}': 1
    }

    input_df = pd.DataFrame([data])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]
    return input_df, education, education_num

# Get user input
df, education, education_num = user_input()

if st.button('Predict'):
    prediction = model.predict(df)[0]
    result = '>50K' if prediction == 1 else '<=50K'

    st.markdown("### \U0001F4BC Hasil Prediksi")
    st.success(f"Predicted Income: {result}")
