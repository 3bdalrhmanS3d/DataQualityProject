import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def restore_original(df, key):
    """Generic function to restore original data"""
    if key in st.session_state:
        return st.session_state[key].copy()
    st.warning(f"No backup found for {key}.")
    return None

def missing_value_analysis(df):
    """Enhanced analysis and presentation of missing values using charts"""
    st.subheader("Missing Values Analysis")
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage.round(2)
    })
    
    # Display summary with improved formatting
    st.write("Missing Values Summary:")
    st.dataframe(missing_summary.style.background_gradient(cmap='YlOrRd'))

    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Bar Plot", "Matrix Plot", "Heatmap"]
    )
    
    fig_size = st.slider("Select Plot Size", 5, 15, 10)
    
    viz_type == "Bar Plot"
    fig, ax = plt.subplots(figsize=(fig_size, fig_size//2))
    msno.bar(df, ax=ax, color="skyblue")
    st.pyplot(fig)
        
    viz_type == "Matrix Plot"
    fig, ax = plt.subplots(figsize=(fig_size, fig_size//2))
    msno.matrix(df, ax=ax)
    st.pyplot(fig)
        
    viz_type == "Heatmap"
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
            st.dataframe(filtered_counts)
        else:
            st.dataframe(value_counts)
        
        # Download button for value counts
        st.download_button(
            label="Download Value Counts CSV",
            data=value_counts.to_csv().encode('utf-8'),
            file_name=f'{selected_column}_value_counts.csv',
            mime='text/csv'
        )

    if df[selected_column].dtype == 'object':
        actions = [
        "Rename Column",
        "Normalize to Lowercase",
        "Replace Specific Values",
        "Convert to Numeric",
        "Visualization",
        "Delete Rows/Columns" ]

    selected_action = st.selectbox("Select Action to Perform:", actions, key=f"action_{selected_column}")

    if selected_action == "Rename Column":
        RenameColumn(df, selected_column)
    elif selected_action == "Normalize to Lowercase":
        NormalizeColumn(df, selected_column)
    elif selected_action == "Replace Specific Values":
        ReplaceSpecificValues(df, selected_column)
    elif selected_action == "Convert to Numeric":
        ConvertToNumeric(df, selected_column)
    elif selected_action == "Visualization":
        Visualization(df, selected_column)
    elif selected_action == "Delete Rows/Columns":
        DeleteRowsColumns(df, selected_column)

def NormalizeColumn(df, selected_column):
    st.session_state["data"] = df
    st.subheader(f"Normalize Column: {selected_column}")
    st.write("### Normalization Options")

    apply_strip = st.checkbox("Remove leading and trailing spaces (strip)", value=True, key=f"strip_{selected_column}")
    apply_lowercase = st.checkbox("Convert to lowercase", value=True, key=f"lowercase_{selected_column}")
    apply_replace_char = st.checkbox("Replace specific character", value=False, key=f"replace_char_{selected_column}")
    apply_replace_spaces = st.checkbox("Replace spaces", value=False, key=f"replace_spaces_{selected_column}")
    apply_remove_all_spaces = st.checkbox("Remove all spaces from text", value=False, key=f"remove_all_spaces_{selected_column}")

    char_to_replace = None
    replacement_character = None
    if apply_replace_char:
        char_to_replace = st.text_input("Enter character to replace:", key=f"char_to_replace_{selected_column}")
        replacement_character = st.text_input("Enter replacement character:", key=f"char_replacement_{selected_column}")

    space_replacement = "_"
    if apply_replace_spaces:
        space_replacement = st.text_input("Enter the character to replace spaces with (default: _):", value="_", key=f"replacement_char_{selected_column}")

    if st.button("Apply Normalization (Preview Only)", key=f"apply_normalization_{selected_column}"):
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
                df[selected_column] = df[selected_column].str.strip()
            if apply_lowercase:
                df[selected_column] = df[selected_column].str.lower()
            if apply_replace_char and char_to_replace and replacement_character:
                df[selected_column] = df[selected_column].str.replace(char_to_replace, replacement_character, regex=False)
            if apply_replace_spaces and space_replacement:
                df[selected_column] = df[selected_column].str.replace(" ", space_replacement, regex=False)
            if apply_remove_all_spaces:
                df[selected_column] = df[selected_column].str.replace(" ", "", regex=False)

            st.session_state["data"] = df
            st.success(f"Normalization changes saved for column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Saving):")
            st.write(df[selected_column].value_counts(dropna=False))
        except Exception as e:
            st.error(f"An error occurred during saving: {e}")

    if st.button("Restore Original Values", key=f"restore_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values for column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Restoration):")
            st.write(df[selected_column].value_counts())
        else:
            st.warning("No original values saved to restore.")

def HandleNumericColumn(df, selected_column):
    st.subheader(f"Handling Numeric Column: {selected_column}")

    # Define actions
    actions = ["Rename Column", "Handle Outliers", "Visualization", "Group By Two Columns"]
    selected_action = st.selectbox("Select Action for Numeric Column", actions, key=f"numeric_action_{selected_column}")

    # Action: Rename Column
    if selected_action == "Rename Column":
        RenameColumn(df, selected_column)

    # Action: Handle Outliers
    elif selected_action == "Handle Outliers":
        HandleOutliers(df, selected_column)

    # Action: Visualization
    elif selected_action == "Visualization":
        Visualization(df, selected_column)

    # Action: Group By Two Columns
    elif selected_action == "Group By Two Columns":
        GroupByTwoColumns(df, selected_column)

def ReplaceSpecificValues(df, selected_column):
    st.session_state["data"] = df
    unique_values = df[selected_column].unique()
    replace_from = st.selectbox("Select value to replace:", unique_values, key=f"replace_from_{selected_column}")
    replace_to = st.text_input("Replace with:", key=f"replace_to_{selected_column}")

    if st.button("Save and Apply Replacement", key=f"save_apply_replace_{selected_column}"):
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
            df[selected_column] = df[selected_column].replace(replace_from, replace_to)
            st.session_state["data"] = df
            st.success(f"Replaced '{replace_from}' with '{replace_to}' in column '{selected_column}'.")

            st.write("Value Frequencies (After Replacement):")
            fig, ax = plt.subplots()
            df[selected_column].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Value Frequencies for Column: {selected_column} (After Replacement)")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.write("Unique Values and Frequencies (After Replacement):")
            st.write(df[selected_column].value_counts())
        else:
            st.error("Please provide both 'Replace from' and 'Replace with' values.")

    if st.button("Restore Original Values", key=f"restore_replace_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values in column '{selected_column}'.")
            st.write("Unique Values and Frequencies (After Restoration):")
            st.write(df[selected_column].value_counts())
        else:
            st.warning("No original values saved to restore.")

def ConvertToNumeric(df, selected_column):
    st.session_state["data"] = df

    if st.button("Save and Convert to Numeric", key=f"save_convert_numeric_{selected_column}"):
        if f"original_{selected_column}" not in st.session_state:
            st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

        try:
            df[selected_column] = pd.to_numeric(df[selected_column], errors="coerce")
            st.session_state["data"] = df
            st.success(f"Converted column '{selected_column}' to numeric.")
            st.write("Column Statistics (After Conversion):")
            st.write(df[selected_column].describe())
        except Exception as e:
            st.error(f"Error converting to numeric: {e}")

    if st.button("Restore Original Values", key=f"restore_numeric_{selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            df[selected_column] = st.session_state[f"original_{selected_column}"]
            st.session_state["data"] = df
            st.success(f"Restored original values in column '{selected_column}'.")
        else:
            st.warning("No original values saved to restore.")

def RenameColumn(df, selected_column):
    new_column_name = st.text_input(f"Enter new name for column '{selected_column}':", key=f"rename_{selected_column}")
    if st.button("Rename Column", key=f"apply_rename_{selected_column}"):
        if new_column_name:
            df.rename(columns={selected_column: new_column_name}, inplace=True)
            st.session_state["data"] = df
            st.success(f"Column '{selected_column}' has been renamed to '{new_column_name}'.")
        else:
            st.error("Please provide a new column name.")

def HandleOutliers(df, selected_column):
    st.subheader("Handle Outliers")

    # Initial Visualization
    st.write("Data Distribution Before Filtering:")
    fig_before, ax_before = plt.subplots()
    df[selected_column].plot(kind='box', ax=ax_before)
    ax_before.set_title(f"Box Plot Before Filtering for {selected_column}")
    st.pyplot(fig_before)

    # Input for Custom Range with adjusted bounds
    min_value = st.number_input(
        f"Enter minimum value to keep for {selected_column}:",
        value=df[selected_column].min() if not df[selected_column].isnull().all() else 0.0,
    )
    max_value = st.number_input(
        f"Enter maximum value to keep for {selected_column}:",
        value=df[selected_column].max() if not df[selected_column].isnull().all() else 0.0,
    )

    if st.button(f"Preview Filtered Data for {selected_column}"):
        if f"original_{selected_column}" not in st.session_state:
            st.session_state[f"original_{selected_column}"] = df.copy()

        df_filtered = df[(df[selected_column] >= min_value) & (df[selected_column] <= max_value)].dropna()

        st.session_state[f"preview_filtered_{selected_column}"] = df_filtered

        st.success("Preview of filtered data based on custom range!")
        st.write("Data Statistics After Filtering:")
        st.write(df_filtered.describe())

        st.write("Filtered Data Preview:")
        st.dataframe(df_filtered)

        # Visualization After Filtering
        st.write("Data Distribution After Filtering:")
        fig_after, ax_after = plt.subplots()
        df_filtered[selected_column].plot(kind='box', ax=ax_after)
        ax_after.set_title(f"Box Plot After Filtering for {selected_column}")
        st.pyplot(fig_after)

    if st.button(f"Save Filtered Data for {selected_column}"):
        if f"preview_filtered_{selected_column}" in st.session_state:
            st.session_state[f"filtered_data_{selected_column}"] = st.session_state[f"preview_filtered_{selected_column}"]
            st.success(f"Filtered data for column '{selected_column}' has been saved.")

    if st.button(f"Restore Original Data for {selected_column}"):
        if f"original_{selected_column}" in st.session_state:
            restore_column(df, selected_column)
            
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
                st.dataframe(rows_to_delete)

                with st.expander("Visualization Before Deletion"):
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_title(f"Value Frequencies for Column: {selected_column} (Before Deletion)")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Error finding rows with value: {e}")

        if st.button("Delete Rows", key=f"delete_rows_{selected_column}"):
            try:
                if "original_data" not in st.session_state:
                    st.session_state["original_data"] = df.copy()

                df = df[df[selected_column] != value_to_delete]
                st.session_state["data"] = df
                st.success(f"Rows where {selected_column} equals '{value_to_delete}' have been deleted.")

                with st.expander("Visualization After Deletion"):
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="bar", ax=ax, color="orange")
                    ax.set_title(f"Value Frequencies for Column: {selected_column} (After Deletion)")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error deleting rows: {e}")

    elif selected_delete_option == "Delete Column":
        st.write("### Values in Column Before Deletion:")
        st.dataframe(df[selected_column].head(10))

        if st.button("Delete Column", key=f"delete_column_{selected_column}"):
            if "original_data" not in st.session_state:
                st.session_state["original_data"] = df.copy()

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
    else:
        st.warning(f"No backup found for column '{selected_column}'. Make sure the column was modified.")

def Visualization(df, selected_column):
    col_type = df[selected_column].dtype

    # Determine visualization options based on column type
    if col_type == 'object':
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
