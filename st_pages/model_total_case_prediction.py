import pickle
import pandas as pd
import streamlit as st
from preprocessing import preprocess_differencing

def load_model_total_case():
    with open('assets/model/xgb_model_total_imputed_cases.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def total_case_prediction_page(mod):
    col1, col2, col3 = mod.columns(3)

    with col1:
        fullyVaccinated = st.number_input("**fullyVaccinated**", min_value=9327654, step=1, help="Number of individuals who have completed the full vaccination regimen for COVID-19")
        new_deaths_smoothed = st.number_input("**new_deaths_smoothed**", min_value=0.0, help="New deaths attributed to COVID-19 (7-day smoothed). Counts can include probable deaths, where reported.")
        new_people_vaccinated_smoothed = st.number_input("**new_people_vaccinated_smoothed**", min_value=0.0, help="Daily number of people receiving their first vaccine dose(7-day smoothed)")
        new_vaccinations_smoothed = st.number_input("**new_vaccinations_smoothed**", min_value=0.0, help="New COVID-19 vaccination doses administered (7-day smoothed)")
        partiallyVaccinated = st.number_input("**partiallyVaccinated**", min_value=4663827, step=1, help="Number of individuals who have received at least one dose of a COVID-19 vaccine but have not yet completed the full vaccination regimen.")
    
    with col2:  
        totalTests = st.number_input("**totalTests**", min_value=4166833, help="Total number of tests for COVID-19")
        totalVaccinations = st.number_input("**totalVaccinations**", min_value=9982068, step=1, help="Total number of COVID-19 vaccination doses administered")
        vaccinated24hours = st.number_input("**vaccinated24hours**", min_value=0.0, help="Number of people vaccinated within a 24-hour period")
        rfh = st.number_input("**rfh**", min_value=0.0, step=0.001, help="10 day rainfall in mm")
        stringency_index = st.number_input("**stringency_index**", min_value=0.0, max_value=100.0, help="Government response composite measure based on 9 response indicators including school/workplace closures,and travel bans, value from 0 to 100(100=strictest)")
        
    with col3:
        test24hours = st.number_input("**test24hours**", min_value=0.0, help="Number of tests conducted in the last 24 hours")
        r3h = st.number_input("**r3h**", min_value=0.0, step=0.001, help="Rainfall 1-month rolling aggregation long term average in mm")
        month = st.number_input("**month**", min_value=1, max_value=12, help="The month in the year with January=1, December=12")
        day_of_week = st.number_input("**day_of_week**", min_value=0, max_value=6, help="The day of the week with Monday=0, Sunday=6")
        
        st.write("<br>", unsafe_allow_html=True)
        predict = st.button("Predict", use_container_width=True)
        
    mod.markdown("🛈 The non-stationary features are differenced to make the data stationary.")
    mod.divider()
    
    input_df = pd.DataFrame({
            'fullyVaccinated': [fullyVaccinated],
            'new_deaths_smoothed': [new_deaths_smoothed],
            'new_people_vaccinated_smoothed': [new_people_vaccinated_smoothed],
            'new_vaccinations_smoothed': [new_vaccinations_smoothed],
            'partiallyVaccinated': [partiallyVaccinated],
            'stringency_index' : [stringency_index],
            'test24hours': [test24hours],
            'totalTests': [totalTests],
            'totalVaccinations': [totalVaccinations],
            'vaccinated24hours': [vaccinated24hours],
            'rfh': [rfh],
            'r3h': [r3h],
            'month':[month],
            'day_of_week': [day_of_week],
            })

    differenced_features = {
        'fullyVaccinated': 9327654, 'partiallyVaccinated': 4663827, 'totalTests': 239937354,
        'totalVaccinations': 9982068
    }
    
    differencing_data = pd.DataFrame([differenced_features])
    preprocessed_data = preprocess_differencing(input_df, differencing_data)

    if predict:
        try:
            model = load_model_total_case()
            # st.write("**You have submitted the following data.**")
            # st.write(input_df)
            prediction = model.predict(preprocessed_data)
            mod.success(f"Predicted Total Imputed Cases: {prediction[0]: .3f}")
        except Exception as e:
            mod.error(f"An error occurred: {e}")