import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import ttest_ind
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from PredictionManager import * 
@dataclass
class AnalysisConfig:
    features_to_include: list
    test_type: str
    significance_level: float = 0.05

def restore_original(df, key):
    """Generic function to restore original data"""
    if key in st.session_state:
        return st.session_state[key].copy()
    st.warning(f"No backup found for {key}.")
    return None

def log_change(operation: str, before: str, details: str):
    """Log a change operation with timestamp"""
    if 'change_log' not in st.session_state:
        st.session_state.change_log = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.change_log.append({
        'timestamp': timestamp,
        'operation': operation,
        'Before': before,
        'details': details
    })

# Add this function to display changes
def show_change_log():
    """Display the change log in the sidebar"""
    if 'change_log' in st.session_state and st.session_state.change_log:
        st.subheader("Change Log")
        view_option = st.radio(
            "View changes:",
            ["Latest First", "Oldest First"]
        )
        
        changes = st.session_state.change_log.copy()
        if view_option == "Latest First":
            changes.reverse()
            
        for change in changes:
            with st.expander(f"{change['operation']} - {change['timestamp']}"):
                st.write(change['Before'])
                st.write(change['details'])
    else:
        st.info("No changes logged yet")

def missing_value_analysis(df):
    """Enhanced analysis and presentation of missing values using charts"""
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    non_missing_values = len(df) - missing_values
    non_missing_percentage = 100 - missing_percentage
    
    # Create a summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage.round(2),
        'Non-Missing Values': non_missing_values,
        'Percentage Non-Missing': non_missing_percentage.round(2)
    })
    
    # Display summary with improved formatting
    st.write("Missing Values Summary:")
    st.table(missing_summary.style.background_gradient(cmap='YlOrRd'))

    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        [ "Stacked Bar Chart", "Bar Plot", "Matrix Plot", "Heatmap"]
    )
    ########################################
    fig_size = st.slider("Select Plot Size", 5, 15, 10)
    viz_type == "Stacked Bar Chart"
    st.write("### Stacked Bar Chart")
    st.write("A stacked bar chart showing missing and non-missing values for each column.")
    
    # Prepare data for stacked bar chart
    plot_data = pd.DataFrame({
        'Missing': missing_values,
        'Non-Missing': non_missing_values
    }, index=df.columns)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size // 2))
    plot_data.plot(kind='bar', stacked=True, ax=ax, color=["orange", "green"])
    ax.set_title("Missing vs Non-Missing Values")
    ax.set_ylabel("Count")
    ax.legend(["Missing", "Non-Missing"])
    st.pyplot(fig)
    ########################################
    viz_type == "Bar Plot"
    st.write("### Bar Plot")
    st.write("A bar plot shows the number of missing values for each column.")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size//2))
    msno.bar(df, ax=ax, color="skyblue")
    st.pyplot(fig)
    ########################################
    viz_type == "Matrix Plot"
    st.write("### Matrix Plot")
    st.write("A matrix plot shows the pattern of missing values in the dataset.")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size//2))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)
    ########################################
    viz_type == "Heatmap"
    st.write("### Heatmap")
    st.write("A heatmap shows the presence of missing values in the dataset, with colors indicating the missingness.")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size//2))
    sns.heatmap(df.isnull(), cbar=True, cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

def handle_object_column(df, selected_column):
    st.write(f"### Analysis for Column: {selected_column}")
    
    # Save original state if not already saved
    if f"original_{selected_column}" not in st.session_state:
        st.session_state[f"original_{selected_column}"] = df[selected_column].copy()
    
    # Enhanced value counts display with search
    with st.expander("View Unique Values"):
        search_term = st.text_input("Search values", key=f"search_{selected_column}")
        value_counts = df[selected_column].value_counts(dropna=False)
        
        if search_term:
            filtered_counts = value_counts[value_counts.index.str.contains(search_term, na=False, case=False)]
            st.table(filtered_counts)
        else:
            st.write(f"Unique Values and Frequencies: { value_counts.count() }")   
            st.table(value_counts)

    if df[selected_column].dtype == 'object':
        actions = [
        "Rename Column",
        "Normalize",
        "Replace Specific Values",
        "Convert to Numeric",
        "Transform",
        "Visualization",
        "Delete Rows/Columns" ]

    action_descriptions = {
        "Rename Column": "Allows you to rename the selected column.",
        "Normalize": "Scales the data in the selected column to a standard range (e.g., 0 to 1).",
        "Replace Specific Values": "Replaces specific values in the selected column with new values.",
        "Convert to Numeric": "Converts the selected column to a numeric format, handling non-numeric entries.",
        "Transform": "Applies transformations like logarithmic, square root, or custom operations to the column.",
        "Visualization": "Generates visualizations for the data in the selected column (e.g., histograms, line charts).",
        "Delete Rows/Columns": "Deletes specific rows or columns based on your selection."
    }
    selected_action = st.selectbox(
        f"Select Action to Perform for {selected_column}:",
        actions,
        key=f"action_{selected_column}_unique"
    )
    description = action_descriptions.get(selected_action, "No description available for this action.")
    st.write(f"**Description:** {description}")

    if selected_action == "Rename Column":
        RenameColumn(df, selected_column)
    elif selected_action == "Normalize":
        NormalizeColumn(df, selected_column)
    elif selected_action == "Replace Specific Values":
        ReplaceSpecificValues(df, selected_column)
    elif selected_action == "Convert to Numeric":
        ConvertToNumeric(df, selected_column)
    elif selected_action == "Transform":
        handle_transformations(df, selected_column)
    elif selected_action == "Visualization":
        Visualization(df, selected_column)
    elif selected_action == "Delete Rows/Columns":
        DeleteRowsColumns(df, selected_column)

def NormalizeColumn(df, selected_column):
    st.session_state["data"] = df
    preview_df =df.copy()
    st.subheader(f"Normalize Column: {selected_column}")
    st.write("### Normalization Options")

    st.markdown("**Normalization allows you to clean and standardize the values in the selected column.** Below are the available options:")
    apply_strip = st.checkbox(
        "Remove leading and trailing spaces (strip)",
        value=False,
        key=f"strip_{selected_column}",
        help="This option removes any spaces at the beginning or end of the text in the column."
    )
    apply_lowercase = st.checkbox(
        "Convert to lowercase",
        value=False,
        key=f"lowercase_{selected_column}",
        help="This option converts all text in the column to lowercase."
    )
    apply_uppercase = st.checkbox(
        "Convert to uppercase",
        value=False,
        key=f"uppercase_{selected_column}",
        help="This option converts all text in the column to uppercase."
    )
    apply_replace_char = st.checkbox(
        "Replace specific character",
        value=False,
        key=f"replace_char_{selected_column}",
        help="This option allows you to replace a specific character in the column with another character."
    )
    apply_replace_spaces = st.checkbox(
        "Replace spaces",
        value=False,
        key=f"replace_spaces_{selected_column}",
        help="This option replaces spaces in the column with a character of your choice."
    )
    apply_remove_all_spaces = st.checkbox(
        "Remove all spaces from text",
        value=False,
        key=f"remove_all_spaces_{selected_column}",
        help="This option removes all spaces from the text in the column."
    )

    char_to_replace = None
    replacement_character = None
    if apply_replace_char:
        char_to_replace = st.text_input("Enter character to replace:", key=f"char_to_replace_{selected_column}")
        replacement_character = st.text_input("Enter replacement character:", key=f"char_replacement_{selected_column}")

    space_replacement = "_"
    if apply_replace_spaces:
        space_replacement = st.text_input("Enter the character to replace spaces with (default: _):", value="_", key=f"replacement_char_{selected_column}",help="Specify the character that will replace spaces in the column.")

    # Create a preview without altering the actual data
    preview_df = df.copy()
    try:
        if apply_strip:
            preview_df[selected_column] = preview_df[selected_column].str.strip()
        if apply_lowercase:
            preview_df[selected_column] = preview_df[selected_column].str.lower()
        if apply_replace_char and char_to_replace and replacement_character:
            preview_df[selected_column] = preview_df[selected_column].str.replace(char_to_replace, replacement_character, regex=False)
        if apply_replace_spaces and space_replacement:
            preview_df[selected_column] = preview_df[selected_column].str.replace(" ", space_replacement, regex=False)
        if apply_remove_all_spaces:
            preview_df[selected_column] = preview_df[selected_column].str.replace(" ", "", regex=False)
        if apply_uppercase:
            preview_df[selected_column] = preview_df[selected_column].str.upper()
        st.write("Unique Values and Frequencies (Preview After Normalization):")
        st.write(preview_df[selected_column].value_counts(dropna=False))
    except Exception as e:
        st.error(f"An error occurred during normalization: {e}")

    if st.button("Save Changes", key=f"save_{selected_column}"):
        if f"original_{selected_column}" not in st.session_state:
            st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

        # Apply changes and save
        try:
            if apply_strip:
                log_change(f"Remove leading and trailing spaces (strip) {selected_column}" , df[selected_column],preview_df[selected_column])
                df[selected_column] = df[selected_column].str.strip()

            if apply_lowercase:
                log_change(f"Convert to lowercase {selected_column}" , df[selected_column],preview_df[selected_column])
                df[selected_column] = df[selected_column].str.lower()
            if apply_replace_char and char_to_replace and replacement_character:
                log_change(f"Replace specific character {selected_column} ", df[selected_column] ,preview_df[selected_column] )
                df[selected_column] = df[selected_column].str.replace(char_to_replace, replacement_character, regex=False)
            if apply_replace_spaces and space_replacement:
                log_change(f"Replace spaces {selected_column}", df[selected_column] ,preview_df[selected_column])
                df[selected_column] = df[selected_column].str.replace(" ", space_replacement, regex=False)
            if apply_remove_all_spaces:
                log_change(f"Remove all spaces from text  {selected_column}", df[selected_column] ,preview_df[selected_column])
                df[selected_column] = df[selected_column].str.replace(" ", "", regex=False)
            if apply_uppercase:
                log_change(f"Convert to uppercase  {selected_column}", df[selected_column] ,preview_df[selected_column])
                df[selected_column] = df[selected_column].str.upper()

            st.session_state["data"] = df
            st.success(f"Normalization changes saved for column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Saving):")
            st.table(df[selected_column].value_counts(dropna=False))
            
        except Exception as e:
            st.error(f"An error occurred during saving: {e}")

    if st.button("Restore Original Values", key=f"restore_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values for column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Restoration):")
            st.table(df[selected_column].value_counts())
            log_change(f"Restore Original Values  {selected_column}" ,preview_df[selected_column], df[selected_column])
        else:
            st.warning("No original values saved to restore.")

def HandleNumericColumn(df, selected_column):
    st.subheader(f"Handling Numeric Column: {selected_column}")

    # Define actions
    actions = ["Handel Missing","Rename Column",  "Handle Outliers","Visualization", "Group By Two Columns","Delete Rows/Columns"]
    selected_action = st.selectbox("Select Action for Numeric Column", actions, key=f"numeric_action_{selected_column}")

    # Action: Rename Column
    if selected_action == "Rename Column":
        RenameColumn(df, selected_column)

    # Action: Visualization
    elif selected_action == "Visualization":
        Visualization(df, selected_column)
    
    elif selected_action == "Handle Outliers":
        HandleOutliers(df, selected_column)
    
    elif selected_action == "Delete Rows/Columns":
        DeleteRowsColumns(df, selected_column)
    # Action: Group By Two Columns
    elif selected_action == "Group By Two Columns":
        GroupByTwoColumns(df, selected_column)
    
    elif selected_action == "Handel Missing":
        replace_column_values(df, selected_column)

def ReplaceSpecificValues(df, selected_column):
    st.session_state["data"] = df
    unique_values = df[selected_column].unique()
    replace_from = st.selectbox("Select value to replace:", unique_values, key=f"replace_from_{selected_column}")
    replace_to = st.text_input("Replace with:", key=f"replace_to_{selected_column}")

    if replace_from and replace_to:
        if f"original_{selected_column}" not in st.session_state:
            st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

        st.write("Value Frequencies (Before Replacement):")
        fig, ax = plt.subplots()
        df[selected_column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Value Frequencies for Column: {selected_column} (Before Replacement)")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Apply replacement
        preview_df = df.copy()
        preview_df[selected_column] = preview_df[selected_column].replace(replace_from, replace_to)

        st.write("Value Frequencies (After Replacement):")
        fig, ax = plt.subplots()
        preview_df[selected_column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Value Frequencies for Column: {selected_column} (After Replacement)")
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("Unique Values and Frequencies (After Replacement):")
        st.write(preview_df[selected_column].value_counts())

        if st.button("Save and Apply Replacement", key=f"save_apply_replace_{selected_column}"):
            log_change(f"Replace Specific Values  {selected_column}" , df[selected_column] ,preview_df[selected_column])
            df[selected_column] = preview_df[selected_column]
            st.session_state["data"] = df
            st.success(f"Replaced '{replace_from}' with '{replace_to}' in column '{selected_column}'.")
    else:
        st.error("Please provide both 'Replace from' and 'Replace with' values.")

    if st.button("Restore Original Values", key=f"restore_replace_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values in column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Restoration):")
            st.write(df[selected_column].value_counts())
            log_change(f"Restore Original Values  {selected_column}" ,preview_df[selected_column], df[selected_column])
        else:
            st.warning("No original values saved to restore.")

def ConvertToNumeric(df, selected_column):
    st.subheader(f"Convert Column to Numeric: {selected_column}")
    preview_df = df.copy()
    try:
        preview_df[selected_column] = pd.to_numeric(preview_df[selected_column], errors="coerce")
        st.write("Unique Values and Frequencies (Preview After Conversion):")
        st.write(preview_df[selected_column].value_counts(dropna=False))
    except Exception as e:
        st.error(f"Error converting to numeric: {e}")

    if st.button("Save and Convert to Numeric", key=f"save_fill_{selected_column}"):
        if f"original_{selected_column}" not in st.session_state:
            st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

        try:
            log_change(f"Convert to Numeric  {selected_column}" , df[selected_column],preview_df[selected_column])
            df[selected_column] = pd.to_numeric(df[selected_column].replace(r'[^\d.]', '', regex=True), errors="coerce")
            st.session_state["data"] = df
            st.success(f"Converted column '{selected_column}' to numeric.")
            st.write("Column Statistics (After Conversion):")
            st.table(df[selected_column].describe())
            
        except Exception as e:
            st.error(f"Error converting to numeric: {e}")

    if st.button("Restore Original Values", key=f"restore_numeric_{selected_column}"):
        restore_column(df, selected_column)
        log_change(f"Restore Original Values  {selected_column}" ,preview_df[selected_column], df[selected_column])

def RenameColumn(df, selected_column):
    new_column_name = st.text_input(f"Enter new name for column '{selected_column}':", key=f"rename_{selected_column}")
    if st.button("Rename Column", key=f"apply_rename_{selected_column}"):
        if new_column_name:
            log_change(f"Rename Column {selected_column}",selected_column, new_column_name)
            df.rename(columns={selected_column: new_column_name}, inplace=True)
            st.session_state["data"] = df
            st.success(f"Column '{selected_column}' has been renamed to '{new_column_name}'.")
            log_change("RenameColumn", selected_column, new_column_name)
        else:
            st.error("Please provide a new column name.")

def outlier_analysis(df, column):
    """Identifies and displays outliers using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    st.write(f"Number of outliers in {column}: {len(outliers)}")
    if not outliers.empty:
        st.write(outliers)
        show_outliers_vis = st.checkbox("Show outliers visualization", key='show_outliers_vis')
        if show_outliers_vis:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            sns.scatterplot(x=outliers[column], y=[0]*len(outliers), color='red', marker='o', ax=ax)
            plt.title(f"Box Plot of {column} with Outliers highlighted")
            st.pyplot(fig)
    log_change("outlier_analysis", "N/A", f"Performed outlier analysis on column: {column}")
    return lower_bound, upper_bound

def handle_outliers(df, column, lower_bound, upper_bound, method):
    """Handles outliers based on the selected method."""
    if method == 'clip':
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        st.success(f"Outliers in {column} have been clipped to the defined bounds.")
    elif method == 'drop':
        df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, inplace=True)
        st.success(f"Outliers in {column} have been removed.")
    else:
        st.error("Invalid method for handling outliers.")
    
    log_change("handle_outliers", "N/A", f"Handled outliers in column: {column} using method: {method}")
    return df

def HandleOutliers(df, selected_column):
    st.subheader("Handle Outliers")
    st.session_state["data"] = df

    # Save original data if not already saved
    if f"original_{selected_column}" not in st.session_state:
        st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

    # Initial Visualization
    st.write("Data Distribution Before Filtering:")
    fig_before, ax_before = plt.subplots()
    df[selected_column].plot(kind='box', ax=ax_before)
    ax_before.set_title(f"Box Plot Before Filtering for {selected_column}")
    st.pyplot(fig_before)

    lower_bound, upper_bound = outlier_analysis(df, selected_column)
    if lower_bound is not None and upper_bound is not None:
        outlier_method = st.selectbox("Select Outlier Handling Method", ['clip', 'drop'])

        # Apply outlier handling method
        preview_df = handle_outliers(df.copy(), selected_column, lower_bound, upper_bound, outlier_method)
        st.session_state[f"preview_filtered_{selected_column}"] = preview_df[selected_column].copy()

        st.success("Filtered data based on custom range!")
        st.write("Data Statistics After Filtering:")
        st.write(preview_df.describe())

        st.write("Filtered Data Preview:")
        st.dataframe(preview_df)

        # Visualization After Filtering
        st.write("Data Distribution After Filtering:")
        fig_after, ax_after = plt.subplots()
        preview_df[selected_column].plot(kind='box', ax=ax_after)
        ax_after.set_title(f"Box Plot After Filtering for {selected_column}")
        st.pyplot(fig_after)

    if st.button(f"Save Filtered Data for {selected_column}"):
        if f"preview_filtered_{selected_column}" in st.session_state:
            log_change(f"Handle Outliers {selected_column}", df[selected_column], st.session_state[f"preview_filtered_{selected_column}"])
            df[selected_column] = st.session_state[f"preview_filtered_{selected_column}"].copy()
            st.session_state[f"filtered_data_{selected_column}"] = df[selected_column].copy()
            st.success(f"Filtered data for column '{selected_column}' has been saved.")
            st.session_state["data"] = df

    if st.button("Restore Original Values", key=f"restore_numeric_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values in column '{selected_column}'.")
            log_change(f"Restore Original Values  {selected_column}" ,preview_df[selected_column], df[selected_column])
        else:
            st.warning("No original values saved to restore.")
         
def DeleteRowsColumns(df, selected_column):
    st.write("### Delete Rows/Columns")
    delete_options = ["Delete Rows", "Delete Column"]
    selected_delete_option = st.selectbox("Select Delete Option", delete_options, key=f"delete_option_{selected_column}")

    if selected_delete_option == "Delete Rows":
        value_to_delete = st.text_input(
            f"Enter value to delete rows where {selected_column} equals this value",
            key=f"value_to_delete_{selected_column}"
        )

        if value_to_delete:
            try:
                rows_to_delete = df[df[selected_column] == value_to_delete]
                st.write("### Rows Matching the Value (Preview):")
                st.table(rows_to_delete)

                with st.expander("Visualization Before Deletion"):
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_title(f"Value Frequencies for Column: {selected_column} (Before Deletion)")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                # Preview changes without saving
                preview_df = df[df[selected_column] != value_to_delete]
                st.write("### Data Preview After Deletion:")
                st.table(preview_df)

                with st.expander("Visualization After Deletion"):
                    fig, ax = plt.subplots()
                    preview_df[selected_column].value_counts().plot(kind="bar", ax=ax, color="orange")
                    ax.set_title(f"Value Frequencies for Column: {selected_column} (After Deletion)")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error finding rows with value: {e}")

        if st.button("Save Changes", key=f"save_delete_rows_{selected_column}"):
            try:
                if "original_data" not in st.session_state:
                    st.session_state["original_data"] = df.copy()
                log_change(f"Delete Rows {selected_column}", df, preview_df)
                df = df[df[selected_column] != value_to_delete]
                st.session_state["data"] = df
                st.success(f"Rows where {selected_column} equals '{value_to_delete}' have been deleted.")
            except Exception as e:
                st.error(f"Error deleting rows: {e}")

    elif selected_delete_option == "Delete Column":
        st.write("### Values in Column Before Deletion:")
        st.dataframe(df[selected_column].head(10))

        # Preview changes without saving
        preview_df = df.drop(columns=[selected_column])
        st.write("### Data Preview After Column Deletion:")
        st.dataframe(preview_df)

        if st.button("Save Changes", key=f"save_delete_column_{selected_column}"):
            if "original_data" not in st.session_state:
                st.session_state["original_data"] = df.copy()
            log_change(f"Delete Column {selected_column}", df, preview_df)
            df.drop(columns=[selected_column], inplace=True)
            st.session_state["data"] = df
            st.success(f"Column '{selected_column}' has been deleted.")

    if st.button("Restore Original Data"):
        restore_column(df, selected_column)

def restore_column(df, selected_column):
    if f"original_{selected_column}" in st.session_state:
        df[selected_column] = st.session_state[f"original_{selected_column}"].copy()
        st.session_state["data"] = df
        st.success(f"Restored original data for column '{selected_column}'.")
        log_change("restore_column", "N/A", f"Restored original data for column: {selected_column}")
    else:
        st.warning(f"No backup found for column '{selected_column}'. Make sure the column was modified.")

def Visualization(df, selected_column):
    col_type = df[selected_column].dtype

    # Determine visualization options based on column type
    if col_type in ['object' , 'bool' ]:
        visualizations = [
            "Bar Plot (Frequency)",
            "Pie Chart"
        ]
    else:
        visualizations = [
            "Histogram",
            "Box Plot",
            "Scatter Plot (Choose X-Axis)",
            "Line Chart",
            "Area Chart",
            "Pair Plot"
        ]

    selected_visualization = st.selectbox("Select Visualization Type", visualizations)
    # Provide a list of palette names for selection
    available_palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    selected_palette_name = st.selectbox("Select Color Palette", available_palettes)

    # Generate the selected palette
    color_palette = sns.color_palette(selected_palette_name)
    show_grid = st.checkbox("Show Gridlines", value=True)

    if st.button("Show Visualization"):
        if selected_visualization == "Bar Plot (Frequency)":
            st.write("### Bar Plot (Frequency)")
            fig, ax = plt.subplots()
            df[selected_column].value_counts().plot(kind="bar", ax=ax, color=color_palette)
            ax.set_title(f"Bar Plot for Column: {selected_column}")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            if show_grid:
                ax.grid(True)
            st.pyplot(fig)

        elif selected_visualization == "Pie Chart":
            st.write("### Pie Chart")
            fig, ax = plt.subplots()
            df[selected_column].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90, colors=sns.color_palette(color_palette))
            ax.set_title(f"Pie Chart for Column: {selected_column}")
            ax.set_ylabel("") 
            st.pyplot(fig)

        elif selected_visualization == "Histogram":
            st.write("### Histogram")
            fig, ax = plt.subplots()
            df[selected_column].plot(kind="hist", bins=20, ax=ax, color=color_palette, edgecolor="black")
            ax.set_title(f"Histogram for Column: {selected_column}")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            if show_grid:
                ax.grid(True)
            st.pyplot(fig)

        elif selected_visualization == "Box Plot":
            st.write("### Box Plot")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, y=selected_column, ax=ax, color=color_palette)
            ax.set_title(f"Box Plot for Column: {selected_column}")
            if show_grid:
                ax.grid(True)
            st.pyplot(fig)

        elif selected_visualization == "Scatter Plot (Choose X-Axis)":
            st.write("### Scatter Plot")
            other_columns = [col for col in df.columns if col != selected_column]
            x_axis_column = st.selectbox("Select X-Axis Column", other_columns)

            if x_axis_column:
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=x_axis_column, y=selected_column, ax=ax, color=color_palette)
                ax.set_title(f"Scatter Plot: {selected_column} vs {x_axis_column}")
                if show_grid:
                    ax.grid(True)
                st.pyplot(fig)

        elif selected_visualization == "Line Chart":
            st.write("### Line Chart")
            fig, ax = plt.subplots()
            df[selected_column].plot(kind="line", ax=ax, color=color_palette)
            ax.set_title(f"Line Chart for Column: {selected_column}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            if show_grid:
                ax.grid(True)
            st.pyplot(fig)

        elif selected_visualization == "Area Chart":
            st.write("### Area Chart")
            fig, ax = plt.subplots()
            df[selected_column].plot(kind="area", ax=ax, color=color_palette)
            ax.set_title(f"Area Chart for Column: {selected_column}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Values")
            if show_grid:
                ax.grid(True)
            st.pyplot(fig)

        elif selected_visualization == "Pair Plot":
            st.write("### Pair Plot")
            numerical_cols = df.select_dtypes(include=['number']).columns
            if len(numerical_cols) > 1:
                fig = sns.pairplot(df[numerical_cols], palette=color_palette)
                st.pyplot(fig)
            else:
                st.error("Not enough numerical columns for a pair plot.")

def GroupByTwoColumns(df, selected_column):
    """
    Function to handle grouping by two columns.
    Works for both numeric and object columns.

    Args:
        df (DataFrame): The DataFrame containing the data.
        selected_column (str): The column to group by.
    """
    st.subheader(f"Group By Two Columns: {selected_column}")

    # Get other columns to group by
    other_columns = [col for col in df.columns if col != selected_column]
    groupby_column = st.selectbox("Select Column to Group By", other_columns, key=f"groupby_{selected_column}")

    if groupby_column and st.button("Save and Show Grouped Data", key=f"save_grouped_{selected_column}"):
        # Group by selected_column and groupby_column, and count occurrences
        grouped_df = df.groupby([selected_column, groupby_column]).size().reset_index(name='counts')

        # Display grouped data
        st.write("### Grouped Data")
        st.write(grouped_df)
        log_change("GroupByTwoColumns", "N/A", f"Grouped by column: {selected_column} and {groupby_column}")

# Use namedtuple to provide structured and easy-to-read analysis results.
CorrelationResult = namedtuple('CorrelationResult', ['correlation_matrix', 'features'])
def correlation_analysis(df, target_column):
    """Perform correlation analysis on all columns and user-selected features."""
    st.subheader("Correlation Analysis")
    
    # Preprocess the DataFrame: Convert non-numeric to NaN and drop problematic rows
    df_numeric = df.select_dtypes(include=['float64', 'int64']).copy()
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric to NaN
    
    # Drop rows with all NaN values (optional)
    if df_numeric.isnull().all(axis=1).any():
        st.warning("Rows with all NaN values will be dropped for correlation analysis.")
        df_numeric = df_numeric.dropna(how='all')
    
    # Check if the numeric DataFrame is empty after cleaning
    if df_numeric.empty:
        st.error("No valid numeric data available for correlation analysis after cleaning.")
        return None
    
    # All Columns Correlation
    st.write("### Correlation Matrix for All Columns")
    correlation_matrix_all = df_numeric.corr()  # Calculate correlation for cleaned numeric columns
    st.table(correlation_matrix_all)
    fig_all, ax_all = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', ax=ax_all)
    st.pyplot(fig_all)
    
    # Custom Features Correlation
    st.write("### Custom Correlation Analysis")
    numerical_columns = df_numeric.columns
    selected_features = st.multiselect("Select Features for Correlation Analysis", numerical_columns)
    
    if selected_features:
        st.write("Correlation Matrix for Selected Features")
        correlation_matrix_custom = df_numeric[selected_features].corr()
        st.table(correlation_matrix_custom)
        fig_custom, ax_custom = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix_custom, annot=True, cmap='coolwarm', ax=ax_custom)
        st.pyplot(fig_custom)
        log_change("correlation_analysis", "N/A", f"Performed correlation analysis on target column: {target_column}")
        return CorrelationResult(correlation_matrix=correlation_matrix_custom, features=selected_features)
    else:
        st.warning("Please select at least one feature for custom correlation analysis.")
        return None

FeatureImportanceResult = namedtuple('FeatureImportanceResult', ['importance_scores', 'model_name'])
def feature_importance(df, target_column):
    """Calculate feature importance using RandomForestClassifier or RandomForestRegressor"""
    st.subheader("Feature Importance Analysis")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check the type of target
    target_type = type_of_target(y)
    
    if target_type in ["binary", "multiclass"]:
        model = RandomForestClassifier()
        model_name = "RandomForestClassifier"
    elif target_type in ["continuous", "continuous-multioutput"]:
        model = RandomForestRegressor()
        model_name = "RandomForestRegressor"
    else:
        st.error(f"Unsupported target type: {target_type}")
        return None

    # Fit the model
    model.fit(X, y)
    importance_scores = model.feature_importances_

    # Prepare feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)
    
    st.write("Feature Importance Scores:")
    st.table(feature_importance_df)

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title(f"Feature Importance - {model_name}")
    st.pyplot(fig)
    log_change("feature_importance", "N/A", f"Calculated feature importance for target column: {target_column}")
    return FeatureImportanceResult(importance_scores=dict(zip(X.columns, importance_scores)), model_name=model_name)

StatisticalTestResult = namedtuple('StatisticalTestResult', ['test_statistic', 'p_value', 'null_hypothesis', 'alternative_hypothesis', 'significant'])
def statistical_tests(df, config: AnalysisConfig):
    """Perform statistical tests on selected features"""
    st.subheader("Statistical Tests")
    features = config.features_to_include
    if len(features) != 2:
        st.error("Please select exactly two features for the statistical test.")
        return None
    feature1, feature2 = features
    if config.test_type == "t-test":
        test_statistic, p_value = ttest_ind(df[feature1].dropna(), df[feature2].dropna())
        significant = p_value < config.significance_level
        result = StatisticalTestResult(
            test_statistic=test_statistic,
            p_value=p_value,
            null_hypothesis=f"There is no significant difference between {feature1} and {feature2}.",
            alternative_hypothesis=f"There is a significant difference between {feature1} and {feature2}.",
            significant=significant
        )
        
        # Create a DataFrame for better table display
        results_df = pd.DataFrame({
            'Metric': ['Test Statistic', 'P-Value', 'Significance Level', 'Significant?', 'Null Hypothesis', 'Alternative Hypothesis'],
            'Value': [
                f"{test_statistic:.4f}",
                f"{p_value:.4f}",
                f"{config.significance_level}",
                "Yes" if significant else "No",
                result.null_hypothesis,
                result.alternative_hypothesis
            ]
        })
        
        st.write("Statistical Test Results:")
        st.table(results_df)
        return result
    else:
        st.error("Unsupported test type.")
        return None

class DataTransformation:
    def __init__(self, df):
        self.df = df
        self.original_df = df.copy()
        self.transformers = {}

    def transform_column(self, column, transform_type, params=None):
            """Transform a single column based on specified type"""
            try:
                if transform_type == "one_hot":
                    encoded_cols = pd.get_dummies(self.df[column], prefix=column)
                    # Update the original dataframe with encoded columns
                    self.df = pd.concat([self.df.drop(columns=[column]), encoded_cols], axis=1)
                    st.session_state["data"] = self.df
                    return encoded_cols
                
                elif transform_type == "label":
                    le = LabelEncoder()
                    self.transformers[column] = le
                    return pd.Series(le.fit_transform(self.df[column]), name=column)
                
                elif transform_type == "minmax":
                    scaler = MinMaxScaler()
                    self.transformers[column] = scaler
                    return pd.Series(scaler.fit_transform(self.df[[column]]).flatten(), name=column)
                
                elif transform_type == "standard":
                    scaler = StandardScaler()
                    self.transformers[column] = scaler
                    return pd.Series(scaler.fit_transform(self.df[[column]]).flatten(), name=column)
                
                elif transform_type == "log":
                    return pd.Series(np.log1p(self.df[column]), name=column)
                
            except Exception as e:
                st.error(f"Transformation error: {str(e)}")
                return None
            
def handle_transformations(df, selected_column):
    """Handle transformations for selected column"""
    st.subheader(f"Transform Column: {selected_column}")
    
    # Initialize transformer
    transformer = DataTransformation(df)
    
    # Select transformation type based on data type
    if df[selected_column].dtype == 'object':
        transform_options = ["one_hot", "label"]
    else:
        transform_options = ["minmax", "standard", "log"]
        
    transform_type = st.selectbox(
        "Select Transformation Type",
        transform_options,
        help="Choose the type of transformation to apply"
    )
    
    # Preview button
    if st.button("Preview Transformation"):
        try:
            # Save original state if not already saved
            if f"original_{selected_column}" not in st.session_state:
                st.session_state[f"original_{selected_column}"] = df[selected_column].copy()
            
            preview = transformer.transform_column(selected_column, transform_type)
            
            if preview is not None:
                st.write("Preview of transformed data:")
                st.table(preview.head() + preview.tail()) 
                
                # Show statistics
                st.write("Statistics after transformation:")
                st.dataframe(preview.describe())
        except Exception as e:
            st.error(f"Error during transformation: {str(e)}")
                
        try:
            # Apply transformation button
            if st.button("Apply Transformation"):
                if transform_type == "one_hot":
                    # Drop original and add encoded columns
                    log_change(f"Transform (one_hot) Column {selected_column}", df[selected_column],preview)
                    df.drop(columns=[selected_column], inplace=True)
                    for col in preview.columns:
                        df[col] = preview[col]
                else:
                    log_change(f"Transform (label) Column {selected_column}", df[selected_column],preview)
                    df[selected_column] = preview
                
                st.success(f"Transformation applied to {selected_column}")
                st.session_state["data"] = df
        except Exception as e:
            st.error(f"Error applying transformation: {str(e)}")
                    
    # Restore original button
    if st.button("Restore Original"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.success(f"Restored original values for {selected_column}")
            st.session_state["data"] = df
            log_change(f"Restore Original Values  {selected_column}" ,preview, df[selected_column])

def replace_column_values(df, selected_column):
    
    if f"original_{selected_column}" not in st.session_state:
        st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

    display_column_statistics(df, selected_column)

    action = st.selectbox(
        "Select action to apply:",
        ["Fill with Mean",
        "Fill with Median", 
        "Fill with Mode", 
        "Fill NaN based on Categorical Target"
        ]
    )

    target_col = None
    if action == "Fill NaN based on Categorical Target":
        target_col = st.selectbox("Select Categorical Target Column", df.columns)
        if target_col == selected_column:
            st.warning("Target column cannot be the same as the selected column.")
            return

    if st.button("Preview Changes"):
        try:
            preview_df = df.copy()
            preview_df[selected_column] = fill_missing_values(preview_df, selected_column, action, target_col)
            with st.expander("View Updated Density Plot"):
                fig, ax = plt.subplots()
                sns.kdeplot(preview_df[selected_column].dropna(), ax=ax, fill=True, color="green")
                ax.set_title(f"Updated Density Plot for {selected_column}")
                st.pyplot(fig)
            st.write("Preview of updated values:", preview_df[selected_column].head())
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Apply Changes"):
        try:
            # Update the DataFrame directly
            df[selected_column] = fill_missing_values(df, selected_column, action, target_col)
            # Save to session state
            st.session_state["data"] = df
            st.success(f"Applied {action} for {selected_column}.")
            log_change(f"Fill Missing Values {selected_column}", st.session_state[f"original_{selected_column}"], df[selected_column])
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Restore Original Values", key=f"restore_numeric_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"].copy()
            st.session_state["data"] = df
            st.success(f"Restored original values in column '{selected_column}'.")
            log_change(f"Restore Original Values {selected_column}", df[selected_column], st.session_state[f"original_{selected_column}"])
        else:
            st.warning("No original values saved to restore.")

def fill_missing_values(df, column, method, target_col=None):
    result = df[column].copy()
    
    if method == "Fill with Mean":
        result = result.fillna(result.mean())
    elif method == "Fill with Median":
        result = result.fillna(result.median())
    elif method == "Fill with Mode":
        result = result.fillna(result.mode()[0])
    elif method == "Fill NaN based on Categorical Target" and target_col:
        if target_col == column:
            raise ValueError("Target column cannot be the same as the selected column.")
        grouped = df.groupby(target_col)[column]
        result = grouped.transform(lambda x: x.fillna(x.mean()))
    
    return result

def display_column_statistics(df, column):
    with st.expander("View Unique Values"):
        value_counts = df[column].value_counts(dropna=False)
        st.table(value_counts)

    with st.expander("View Column Density and Statistics"):
        fig, ax = plt.subplots()
        sns.kdeplot(df[column].dropna(), ax=ax, fill=True, color="blue")
        ax.set_title(f"Density Plot for {column}")
        ax.set_xlabel("Values")
        st.pyplot(fig)

        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Mode', 'Missing Values'],
            'Value': [df[column].mean(), df[column].median(), df[column].mode()[0], df[column].isnull().sum()]
        })
        st.table(stats_df)

def HandleBooleanColumn(df, selected_column):
    st.subheader(f"Handling Boolean Column: {selected_column}")

    actions = ["Rename Column", "Replace Specific Values", "Delete Rows/Columns", "Convert to Numeric", "Visualization"]
    selected_action = st.selectbox("Select Action for Boolean Column", actions, key=f"boolean_action_{selected_column}")

    # Save original state if not already saved
    if f"original_{selected_column}" not in st.session_state:
        st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

    if selected_action == "Rename Column":
        RenameColumn(df, selected_column)

    # Action: Replace Specific Values
    elif selected_action == "Replace Specific Values":
        ReplaceSpecificValues(df, selected_column)

    # Action: Delete Rows/Columns
    elif selected_action == "Delete Rows/Columns":
        DeleteRowsColumns(df, selected_column)
    
    elif selected_action == "Convert to Numeric":
        df[selected_column] = df[selected_column].astype(int)
        st.success(f"Converted boolean column '{selected_column}' to numeric.")
        st.session_state["data"] = df
        log_change("ConvertToNumeric", st.session_state[f"original_{selected_column}"].to_string(), df[selected_column].to_string())

    elif selected_action == "Visualization":
        Visualization(df, selected_column)

    log_change("HandleBooleanColumn", df[selected_column].to_string(), f"Handled boolean column: {selected_column}")

    # Restore original button
    if st.button("Restore Original Values", key=f"restore_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values for column '{selected_column}'.")
            log_change("Restore Original Values", "N/A", f"Restored original values for column: {selected_column}")
        else:
            st.warning("No original values saved to restore.")

def Handle_Duplicates():
    df = st.session_state['data']
    st.subheader("Handle Duplicates")
    duplicates = df[df.duplicated()]
    num_duplicates = duplicates.shape[0]
    st.write(f"Number of duplicate rows: {num_duplicates}")

    if num_duplicates > 0:
        st.write("Duplicate rows:")
        st.dataframe(duplicates)

        # Visualization before removal
        st.write("### Data Before Removing Duplicates")
        st.write("Descriptive Statistics:")
        st.table(df.describe())
        st.write("Data Preview:")
        st.dataframe(df.head())

        if st.button("Remove Duplicates"):
            log_change(f"Removed duplicates (Number of duplicate rows: {num_duplicates})", duplicates, df)
            df.drop_duplicates(inplace=True)
            st.session_state['data'] = df  # Save changes to session state
            st.success("Duplicates removed!")
            st.write(f"Number of duplicate rows after removal: {df.duplicated().sum()}")

            # Visualization after removal
            st.write("### Data After Removing Duplicates")
            st.write("Descriptive Statistics:")
            st.table(df.describe())
            st.write("Data Preview:")
            st.dataframe(df.head())

            if st.button("Restore Original Data"):
                if 'original_data' in st.session_state:
                    df = st.session_state['original_data'].copy()
                    st.session_state['data'] = df
                    st.success("Original data restored.")
                    st.write("### Restored Data")
                    st.write("Descriptive Statistics:")
                    st.table(df.describe())
                    st.write("Data Preview:")
                    st.dataframe(df.head())
                else:
                    st.warning("No original data to restore.")