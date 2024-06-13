import warnings
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings('ignore')


file_path = './assets/data/preprocessed_data_updated.csv'

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def cases_by_stringency_index(df):
    fig = px.box(df, x='stringency_index', y='imputed_total_cases',
                title='Cases by Stringency Index',
                color='stringency_index',
                height=360,
                )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig)

def total_deaths_by_stringency_index(df):
    fig = px.box(df, x='stringency_index', y='imputed_total_deaths',
                title='Total deaths by Stringency Index',
                color='stringency_index', height=360)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig)

def deaths_by_reproduction_rate(df):
    fig = px.box(df, x='reproduction_rate', y='imputed_total_deaths',
                title='Deaths by Reproduction Rate',
                color='reproduction_rate', height=360)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig)

# Create a function to plot cases analysis
def plot_cases_analysis(df):
    plot_df_cases = df.groupby(['Unnamed: 0'])['imputed_total_cases'].sum().reset_index()
    fig = px.line(
        plot_df_cases,
        x='Unnamed: 0',
        y='imputed_total_cases',
        title='Total Cases Over Time',
        labels={'Unnamed: 0': 'Date', 'imputed_total_cases': 'Total Cases'}, height=360
    )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(title_text='Total Cases', title_font=dict(size=14))
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_deaths_analysis(df):
    plot_df_deaths = df.groupby(['Unnamed: 0'])['imputed_total_deaths'].sum().reset_index()
    fig = px.line(
        plot_df_deaths,
        x='Unnamed: 0',
        y='imputed_total_deaths',
        title='Total Deaths Over Time',
        labels={'Unnamed: 0': 'Date', 'imputed_total_deaths': 'Total Deaths'}, height=360
    )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(title_text='Total Deaths', title_font=dict(size=16))
    fig.update_layout(
        autosize=True,
        height=360,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_vaccinations_analysis(df):
    plot_df_vaccinations = df.groupby(['Unnamed: 0'])['totalVaccinations'].sum().reset_index()
    fig = px.line(
        plot_df_vaccinations,
        x='Unnamed: 0',
        y='totalVaccinations',
        title='Total Vaccinations Over Time',
        labels={'Unnamed: 0': 'Date', 'totalVaccinations': 'Total Vaccinations'}, height=360
    )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(title_text='Total Vaccinations', title_font=dict(size=16))
    fig.update_layout(title_font=dict(size=24))
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main(eda):
    df = load_data(file_path)
    
    cols_0, _, cols_1 = eda.columns([0.8,1,0.4])
    with cols_0:
        st.write("<h3>📈 Exploratory Data Analysis</h3>", unsafe_allow_html=True)
        st.write("Explore the relationship between various COVID-19 metrics by selecting from the sidebar")
        
    with cols_1:
        df = load_data(file_path)
        st.write("<br>"*2, unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Data', data=csv, file_name="covid_data.csv", mime="text/csv", use_container_width=True, key='eda_download')
        
    cols = eda.columns(3)
    with cols[0]:
        with st.container(border=True):
            page = st.selectbox('Select Analysis Type', ['Cases Analysis', 'Deaths Analysis', 'Vaccinations Analysis'])
            
            if page == 'Cases Analysis':
                plot_cases_analysis(df)

            elif page == 'Deaths Analysis':
                plot_deaths_analysis(df)

            elif page == 'Vaccinations Analysis':
                plot_vaccinations_analysis(df)

    with cols[1]:
        with st.container(border=True):
            cols_ = st.columns(2)
            with cols_[0]:
                num_filter = st.selectbox("Select Numerical Column", ['imputed_total_cases', 'imputed_total_deaths', 'totalVaccinations', 'totalTests'])
            with cols_[1]:
                cat_filter = st.selectbox("Select Categorical Column", ['stringency_index', 'reproduction_rate', 'rfh', 'r3h'])
            
            if num_filter is not None:
                fig = px.scatter(df, x=num_filter, y='totalVaccinations', color=cat_filter, size=num_filter, height=360)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with cols[2]:
        with st.container(border=True):
            plot_choice = st.selectbox("Select a Plot", ["Cases by Stringency Index","Total Deaths Distribution by Stringency index", "Deaths Distribution by Reproduction Rate"])
                
            if plot_choice == "Cases by Stringency Index":
                cases_by_stringency_index(df)
            elif plot_choice == "Deaths Distribution by Reproduction Rate":
                deaths_by_reproduction_rate(df)
            else:
                total_deaths_by_stringency_index(df)

    eda.write("<br>", unsafe_allow_html=True)
    eda.dataframe(df)
    

if __name__ == '__main__':
    st.set_page_config(page_title='COVID-19 Case Prediction App', page_icon='assets/img/favicon.png', layout='wide')
    main(st)