import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import base64
from io import StringIO

def missing_value_analysis(df):
    # Displays the number of missing values per column. 
    missing_values = df.isnull().sum() 
    st.write("Missing Values per Column:")
    st.table(missing_values)
    
    # Missingno bar plot
    st.write("Columns Values Bar Plot:")
    fig, ax = plt.subplots()
    msno.bar(df, ax=ax, color="red")
    st.pyplot(fig)

    # Seaborn heatmap
    st.write("Missing Values Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Bar plot of missing values
    st.write("Missing Values Count Plot:")
    fig, ax = plt.subplots()
    missing_values.plot(kind='bar', color='blue', ax=ax)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Number of Missing Values')
    ax.set_title('Missing Values Count per Column')
    st.pyplot(fig)