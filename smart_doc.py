import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import pickle

with open("sav_file/diabetes_model.sav", "rb") as file:
    diabetes_model = pickle.load(file, encoding="utf-8")
with open("sav_file/heart_model.sav", "rb") as file:
    heart_model = pickle.load(file, encoding="utf-8")
with open("sav_file/cancer_model.sav", "rb") as file:
    cancer_model = pickle.load(file, encoding="utf-8")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://i.gifer.com/origin/b8/b8a2b545a0604c87829290f3753dd9ed_w200.gif');
        background-size: cover;
        background-position: center center;

        opacity: 0.8
    }
    </style>
    """,
    unsafe_allow_html=True
)

#sidebar


with st.sidebar:
    selected = st.selectbox('Multipal disease Prediction system',
                           ['Home','Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Cancer Prediction'],
                           index=0)

#home page
if (selected == 'Home'):
    st.markdown(
        """
        <h1 style="font-size:60px;font-family:Courier New, monospace; color: green; text-align: center ;
        ">📝Quick Checkup📝</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:30px;font-family:Courier New, monospace; color: red; text-align: center;
        ">Welcome to Quick Checkup!</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:20px;font-family:Courier New, monospace; color: white; text-align: left;
        "> * This is a disease prediction system where you can check for diabetes, heart disease, and cancer.</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:20px;font-family:Courier New, monospace; color: white; text-align: left;
        "> * Please provide the required information and select the disease you want to check for from the sidebar.</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:30px;font-family:Courier New, monospace; color: red; text-align: center;
        ">Thank You!</h1>
        """,
        unsafe_allow_html=True
    )
#diabates page
if (selected == 'Diabetes Prediction'):
    st.markdown(
        """
        <h1 style="font-size:50px;font-family:Courier New, monospace; color: green; text-align: center ;
        ">📝Diabetes Checkup📝</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:20px;font-family:Courier New, monospace; color: Red; text-align: center;
        "> Fill with correct details</h1>
        """,
        unsafe_allow_html=True
    )
    st.write("#")
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col3:
        Insulin = st.text_input('Insulin Level')

    with col1:
        BMI = st.text_input('BMI value')

    with col2:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col3:
        Age = st.text_input('Age of the Person')
        # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

#heart page
if (selected == 'Heart Disease Prediction'):
    st.markdown(
        """
        <h1 style="font-size:50px;font-family:Courier New, monospace; color: Greed; text-align: center ;
        ">📝Heart Checkup📝</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:20px;font-family:Courier New, monospace; color: Red; text-align: Center;
        "> Fill with correct details</h1>
        """,
        unsafe_allow_html=True
    )
    st.write("#")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        cp = st.text_input('Chest Pain types')

    with col3:
        trestbps = st.text_input('Resting Blood Pressure')

    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col3:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col1:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col2:
        exang = st.text_input('Exercise Induced Angina')

    with col3:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col2:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col3:
        thal = st.text_input('Thal(0 = normal; 1 = fixed defect; 2 = reversable defect) ')
    age = int(age) if age else None
    cp = int(cp) if cp else None
    trestbps = int(trestbps) if trestbps else None
    chol = int(chol) if chol else None
    fbs = int(fbs) if fbs else None
    restecg = int(restecg) if restecg else None
    thalach = int(thalach) if thalach else None
    exang = int(exang) if exang else None
    oldpeak = float(oldpeak) if oldpeak else None
    slope = int(slope) if slope else None
    ca = int(ca) if ca else None
    thal =int(thal) if thal else None

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict(
            [[age,cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

#cancer page
if (selected == 'Cancer Prediction'):
    st.markdown(
        """
        <h1 style="font-size:50px;font-family:Courier New, monospace; color: green; text-align: center ;
        ">📝Cancer Checkup📝</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <h1 style="font-size:20px;font-family:Courier New, monospace; color: Red; text-align: center;
        "> Fill with correct details</h1>
        """,
        unsafe_allow_html=True
    )
    st.write("#")
    # Create input fields for each feature

    Clump_Thickness = st.number_input("Clump Thickness")
    Cell_Size = st.number_input("Cell Size")
    Cell_shape= st.number_input("Cell Shape")
    Marginal_Adhesion = st.number_input("Marginal Adhesion")
    Epithelial_Cell_Size = st.number_input("Epithelial Cell Size")
    Bare_Nuclei = st.number_input("Bare Nuclei")
    Bland_Chromatin = st.number_input("Bland Chromatin")
    Normal_Nucleoli = st.number_input("Normal Nucleoli")
    Mitoses	= st.number_input("Mitoses")

    # code for Prediction
    cancer_diagnosis = ''

    if st.button('cancer Test Result'):
        prediction = cancer_model.predict(
            [[Clump_Thickness, Cell_Size, Cell_shape, Marginal_Adhesion, Epithelial_Cell_Size,
              Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses ]])

        # Display the prediction
        if prediction[0] == 0:
            st.write("Prediction: Benign")
        else:
            st.write("Prediction: Malignant")

    st.success(cancer_diagnosis)
