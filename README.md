# Data Quality App

This is a Python-based web application built using
**Streamlit** for performing common data quality tasks such as handling missing values, duplicates,
and outliers in datasets, The app also integrates with **Ollama** for a chatbot
interface to interact with the dataset and answer questions using a **Retrieval-Augmented Generation (RAG)** model.

For those who wish to try the app, you can access it [here](https://dataqualityproject.streamlit.app/).


## Features
- **Dataset Upload:** Upload CSV or Excel files.
- **Data Quality Tasks:** Perform the following tasks on the uploaded dataset:
  - Dataset Info: View basic dataset information.
  - Describe Dataset: Get descriptive statistics of the dataset.
  - Handle Missing Values: Fill or drop missing values.
  - Handle Duplicates: Identify and remove duplicate rows.
  - Handle Outliers: Remove outliers using the Interquartile Range (IQR) method.
- **Chat with Dataset:** Ask questions about the dataset using the RAG model powered by Ollama.

## Prerequisites

Before running the project, make sure you have Python 3.12 installed on your system.

## Installation

1. **Clone the repository (optional)**

   ```bash
      git https://github.com/3bdalrhmanS3d/DataQualityProject.git
      cd DataQualityProject
   ```

2. **Create a virtual environment**

   ```bash
      python -m venv venv
   ```

3. **Activate the virtual environment**

   On Windows:
   ```bash
      venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
      source venv/bin/activate
   ```

4. **Install the required dependencies**

   ```bash
      pip install -r requirements.txt
   ```

   Alternatively, install the required libraries manually:

   ```bash
      pip install streamlit pandas ollama
   ```

5. **Verify the installed libraries**

   ```bash
      pip list
   ```

6. **Run the Streamlit app**

   ```bash
      streamlit run RAG.py
   ```

   The app will open in your default web browser.

## Project Structure

- **RAG.py**: The main Streamlit application script that runs the web interface.
- **requirements.txt**: A file listing the project dependencies (optional, for convenience).
- **venv/**: Virtual environment directory (this will be created when you set up the virtual environment).

## Usage

- Upload your dataset (CSV or Excel) via the sidebar.
- Select the task you want to perform from the navigation menu in the sidebar:
  - **Dataset Info**: View basic information about your dataset (columns, types, non-null counts).
  - **Describe Dataset**: View the descriptive statistics of the dataset.
  - **Handle Missing Values**: Choose to fill or drop missing values from columns.
  - **Handle Duplicates**: Identify and remove duplicate rows.
  - **Handle Outliers**: Remove outliers using the IQR method.
  - **Chat using RAG**: Interact with your dataset via a chatbot powered by Ollama.

## Download Modified Dataset

After performing any changes, you can download the modified dataset by clicking the download button on the sidebar.

## Requirements

- **Python 3.12**
- **Streamlit**: For creating the web interface.
- **Pandas**: For data manipulation and analysis.
- **Ollama**: For chatbot integration using the RAG model.

--- 

## requirements.txt

Here are the required libraries for this project:

```txt
   streamlit
   pandas
   ollama
   seaborn
   matplotlib
   missingno
```
