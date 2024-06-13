import pickle
import streamlit as st
from st_pages.model_total_case_prediction import total_case_prediction_page
from st_pages.model_total_death_prediction import total_death_prediction_page

model_path = './assets/model/xgb_model_total_imputed_cases.pkl'

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess(main_dataframe, dataframe_with_last_known_value):
    """
    Preprocess the main dataframe by subtracting values from the dataframe_with_last_known_value.

    Parameters:
    main_dataframe (pd.DataFrame): The main dataframe containing 14 features. 
                                   This dataframe is expected to have one row of user input.
    dataframe_with_last_known_value (pd.DataFrame): The dataframe containing 6 features and their known values.
                                                    This dataframe is expected to have one row with known values.

    Returns:
    pd.DataFrame: A new dataframe with the same structure as the main_dataframe, 
                  where the values of the 6 matching features have been subtracted 
                  by their corresponding values in the dataframe_with_last_known_value.
    """
    # Identify the common columns
    common_columns = main_dataframe.columns.intersection(dataframe_with_last_known_value.columns)

    # Subtract the values of the known features
    for column in common_columns:
        main_dataframe[column] = main_dataframe[column] - dataframe_with_last_known_value[column]
        
    return main_dataframe

# Main function to display the Streamlit app
def main(mod):
    
    cols = mod.columns([1,0.4,0.6])
    with cols[0]:
        st.write("<h3>😷 COVID-19 Case Prediction</h3>", unsafe_allow_html=True)
        st.caption("""Enter the required features to predict the total imputed COVID-19 cases. Please provide the values for the following features:""")
        
    with cols[2]:
        st.write("<br>", unsafe_allow_html=True)
        options = st.selectbox("Select a Prediction Model:", ["Total Death Prediction", "Total Case Prediction"])
        
    mod.caption("""🛈 The minimum values for 'fullyVaccinated', 'partiallyVaccinated', 'totalVaccinations' and 'totalTests' are the last known values as on 21st April 2024.""")
        
    if options == "Total Death Prediction":
        total_death_prediction_page(mod)
    elif options == "Total Case Prediction":
        total_case_prediction_page(mod)