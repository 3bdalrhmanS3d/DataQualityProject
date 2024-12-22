from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


class PredictionManager:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, target_column, features, test_size=0.2):
        """Prepare data for training"""
        X = self.df[features]
        y = self.df[target_column]
        smote = SMOTE(random_state=42)  # Handle class imbalance
        X_resampled, y_resampled = smote.fit_resample(X, y)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
        )


def predict_new_use_case(df):
    """Handle predictions and model evaluation"""
    st.subheader("Machine Learning Model Configuration and Testing")
    pred_manager = PredictionManager(df)

    # Sidebar Configuration
    st.sidebar.write("### Model Configuration")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, step=0.1)
    n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 50, 500, 100, step=50)
    max_depth = st.sidebar.slider("Max Tree Depth (Random Forest)", 3, 20, 10, step=1)

    # Feature and Target Selection
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_column = st.selectbox("Select Target Column", df.columns, help="Target column to predict")
    features = st.multiselect("Select Feature Columns", [col for col in numerical_columns if col != target_column])

    if target_column and features:
        pred_manager.prepare_data(target_column, features, test_size)

        # Compare Models
        if st.button("Compare Models"):
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42)
            }
            results = []
            for model_name, model in models.items():
                model.fit(pred_manager.X_train, pred_manager.y_train)
                y_pred = model.predict(pred_manager.X_test)
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(pred_manager.y_test, y_pred),
                    "F1 Score": f1_score(pred_manager.y_test, y_pred, average="weighted"),
                    "Precision": precision_score(pred_manager.y_test, y_pred, average="weighted"),
                    "Recall": recall_score(pred_manager.y_test, y_pred, average="weighted")
                })

            # Display Model Comparison
            results_df = pd.DataFrame(results)
            st.write("### Model Performance Comparison")
            st.table(results_df)

            # Highlight Best Model
            best_model = results_df.loc[results_df['F1 Score'].idxmax()]
            st.success(f"The best model is {best_model['Model']} with F1 Score: {best_model['F1 Score']:.2f}")

        # Train Single Model
        if st.button("Train Model"):
            with st.spinner("Training Random Forest Model..."):
                try:
                    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    rf_model.fit(pred_manager.X_train, pred_manager.y_train)
                    st.session_state['trained_model'] = rf_model

                    # Evaluate Performance
                    y_pred = rf_model.predict(pred_manager.X_test)
                    st.write("### Model Performance")
                    st.code(classification_report(pred_manager.y_test, y_pred))

                    # Confusion Matrix
                    st.write("### Confusion Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(pred_manager.y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Make Predictions
        if "trained_model" in st.session_state:
            st.write("### Make Predictions on New Data")
            new_data = {}
            for feature in features:
                new_data[feature] = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
            if st.button("Predict"):
                # try:
                #     input_data = pd.DataFrame([new_data])
                #     prediction = st.session_state['trained_model'].predict(input_data)
                #     st.success(f"The prediction for {target_column} is: {prediction[0]}")
                
                try:
                    # Prepare input data
                    input_data = pd.DataFrame([new_data])
                    
                    # Make prediction
                    prediction = st.session_state['trained_model'].predict(input_data)
                    
                    # Show prediction
                    st.success(f"Predicted {target_column}: {prediction[0]}")
                    
                    # Show prediction probability
                    proba = st.session_state['trained_model'].predict_proba(input_data)
                    st.write("### Prediction Probabilities")
                    proba_df = pd.DataFrame(
                        proba,
                        columns=st.session_state['trained_model'].classes_
                    )
                    st.dataframe(proba_df)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
