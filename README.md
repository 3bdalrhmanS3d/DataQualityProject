# Data Quality App

This is a Python-based web application built using
**Streamlit** for performing common data quality tasks such as handling missing values, duplicates,
and outliers in datasets, The app also integrates with **Ollama** for a chatbot
interface to interact with the dataset and answer questions using a **Retrieval-Augmented Generation (RAG)** model.

For those who wish to try the app, you can access it [here](https://dataqualityproject.streamlit.app/).

## Features

### 1. Data Quality Analysis
- **Dataset Upload:** Upload CSV or Excel files.
- **Dataset Info:** View basic dataset information.
- **Describe Dataset:** Get descriptive statistics of the dataset.
- **Handle Missing Values:** Fill or drop missing values.
- **Handle Duplicates:** Identify and remove duplicate rows.
- **Outlier Detection:** Identify and handle outliers.
- **Data Type Conversion:** Convert data types and normalize columns.

### 2. Data Visualization
- **Interactive Plots:** Bar plots, pie charts, histograms, box plots, scatter plots, line charts, area charts, and pair plots.
- **Correlation Matrices:** View correlation between features.
- **Distribution Analysis:** Analyze data distributions.
- **Custom Color Palettes:** Choose from various color palettes for visualizations.

### 3. Machine Learning
- **Model Comparison:** Compare multiple models (Random Forest, SVM, Logistic Regression).
- **Feature Importance:** Analyze feature importance using RandomForestClassifier.
- **Cross-Validation:** Perform cross-validation to evaluate model performance.
- **Model Performance Metrics:** View accuracy, F1 score, precision, and recall.
- **Interactive Prediction Interface:** Make predictions on new data.

### 4. RAG-powered Chat
- **Dataset Querying:** Query the dataset using natural language.
- **Context-Aware Responses:** Get context-aware responses from the dataset.
- **Code Snippet Generation:** Generate code snippets for data analysis.
- **Interactive Chat Interface:** Chat with the dataset using Ollama's RAG model.

## Prerequisites

Before running the project, make sure you have Python 3.12 installed on your system and Ollama (for RAG features).

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
```txt
  DataQualityProject/
  ├── RAG.py                 # Main application
  ├── HandlingSection.py     # Data handling components
  ├── PredictionManager.py   # ML model management
  ├── Lib.py                # Common utilities
  ├── requirements.txt      # Dependencies
  └── README.md            # Documentation
```
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

## Data Processing Features
- Missing Values: Multiple imputation methods
- Outliers: IQR-based detection and handling
- Transformations: Scaling, encoding, normalization
- Feature Engineering: Automated and manual options

## Machine Learning Capabilities
- Models:
    - Random Forest
    - Support Vector Machines
    - Logistic Regression
- Metrics:
    - Accuracy
    - F1 Score
    - Precision
    - Recall
- Visualization:
    - Confusion Matrix
    - ROC Curves
    - Feature Importance

## requirements.txt

Here are the required libraries for this project:

```txt
  streamlit
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  ollama
  missingno
  imbalanced-learn
```
