import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno



def missing_value_analysis(df):
    """Analysis and presentation of lost values using charts"""
    missing_values = df.isnull().sum()
    st.write("Missing Values per Column:")
    st.table(missing_values)

    # Missingno bar plot
    st.write("Columns Values Bar Plot:")
    fig, ax = plt.subplots()
    msno.bar(df, ax=ax, color="red")
    st.pyplot(fig)

    # Bar plot of missing values
    st.write("Missing Values Count Plot:")
    fig, ax = plt.subplots()
    missing_values.plot(kind="bar", color="blue", ax=ax)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    ax.set_title("Missing Values Count per Column")
    st.pyplot(fig)

    # Seaborn heatmap
    st.write("Missing Values Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    


def handle_object_column(df, selected_column):
    col_type = df[selected_column].dtype 

    if col_type in ['int64', 'float64']:
        st.write(f"Column '{selected_column}' is numeric.")
    elif col_type == 'object':
        st.write(f"### Analyzing column: {selected_column}")
        st.write("Unique Values and Frequencies (Before Cleaning):")
        st.write(df[selected_column].value_counts(dropna=False))

         # خيارات الإجراء
        actions = [
            "Normalize to lowercase",
            "Replace Specific Values",
            "Convert to Numeric",
            "Visualization",
        ]
        
        selected_action = st.selectbox("Select Action", actions, key=f"action_{selected_column}")

        if selected_action == "Normalize to lowercase":
            # Normalize to lowercase
            if st.button("Apply", key=f"normalize_{selected_column}"):
                if f"original_{selected_column}" not in st.session_state:
                    st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

                df[selected_column] = (
                    df[selected_column]
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                )
                st.session_state["data"] = df
                st.success(f"Normalized values in column '{selected_column}'.")
                st.write("Unique Values and Frequencies (After Normalization):")
                st.write(df[selected_column].value_counts(dropna=False))

                # Restore original values
                if st.button("Restore Original Values", key=f"restore_{selected_column}"):
                    if f"original_{selected_column}" in st.session_state:
                        df[selected_column] = st.session_state[f"original_{selected_column}"]
                        st.session_state["data"] = df
                        st.success(f"Restored original values in column '{selected_column}'.")
                        st.write("Unique Values and Frequencies (After Restoration):")
                        st.write(df[selected_column].value_counts())
                    else:
                        st.warning("No original values saved to restore.")

        elif selected_action == "Replace Specific Values":
            # Replace specific values
            st.write("### Replace Specific Values")
            unique_values = df[selected_column].unique()
            replace_from = st.selectbox("Select value to replace:", unique_values, key=f"replace_from_{selected_column}")
            replace_to = st.text_input("Replace with:", key=f"replace_to_{selected_column}")

            if st.button("Apply Replacement", key=f"apply_replace_{selected_column}"):
                if replace_from and replace_to:
                    if f"original_{selected_column}" not in st.session_state:
                        st.session_state[f"original_{selected_column}"] = df[selected_column].copy()

                    # Display frequency plot before replacement
                    st.write("Value Frequencies (Before Replacement):")
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="bar", ax=ax)
                    ax.set_title(f"Value Frequencies for Column: {selected_column} (Before Replacement)")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                    df[selected_column] = df[selected_column].replace(replace_from, replace_to)
                    st.session_state["data"] = df
                    st.success(f"Replaced '{replace_from}' with '{replace_to}' in column '{selected_column}'.")

                    # Display frequency plot after replacement
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

            # Restore original replaced values
            if st.button("Restore Original Replaced Values", key=f"restore_replace_{selected_column}"):
                if f"original_{selected_column}" in st.session_state:
                    df[selected_column] = st.session_state[f"original_{selected_column}"]
                    st.session_state["data"] = df
                    st.success(f"Restored original values in column '{selected_column}'.")
                    st.write("Unique Values and Frequencies (After Restoration):")
                    st.write(df[selected_column].value_counts())
                else:
                    st.warning("No original values saved to restore.")
        elif selected_action == "Convert to Numeric":
            # Convert to numeric
            if st.button("Convert to Numeric", key=f"convert_numeric_{selected_column}"):
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

            # Restore original numeric column
            if st.button("Restore Original Numeric Values", key=f"restore_numeric_{selected_column}"):
                if f"original_{selected_column}" in st.session_state:
                    df[selected_column] = st.session_state[f"original_{selected_column}"]
                    st.session_state["data"] = df
                    st.success(f"Restored original values in column '{selected_column}'.")
                else:
                    st.warning("No original values saved to restore.")

        elif selected_action == "Visualization":
            if col_type == 'object':
                visualizations = [
                    "Bar Plot (Frequency)",
                    "Pie Chart"
                ]
            else:
                visualizations = [
                    "Histogram",
                    "Box Plot",
                    "Scatter Plot (Choose X-Axis)"
                ]
            
            selected_visualization = st.selectbox("Select Visualization Type", visualizations)

            if st.button("Show Visualization"):
                if selected_visualization == "Bar Plot (Frequency)":
                    st.write("### Bar Plot (Frequency)")
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_title(f"Bar Plot for Column: {selected_column}")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                elif selected_visualization == "Pie Chart":
                    st.write("### Pie Chart")
                    fig, ax = plt.subplots()
                    df[selected_column].value_counts().plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                    ax.set_title(f"Pie Chart for Column: {selected_column}")
                    ax.set_ylabel("") 
                    st.pyplot(fig)

                elif selected_visualization == "Histogram":
                    st.write("### Histogram")
                    fig, ax = plt.subplots()
                    df[selected_column].plot(kind="hist", bins=20, ax=ax, color="orange", edgecolor="black")
                    ax.set_title(f"Histogram for Column: {selected_column}")
                    ax.set_xlabel("Values")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                elif selected_visualization == "Box Plot":
                    st.write("### Box Plot")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df, y=selected_column, ax=ax, color="green")
                    ax.set_title(f"Box Plot for Column: {selected_column}")
                    st.pyplot(fig)

                elif selected_visualization == "Scatter Plot (Choose X-Axis)":
                    st.write("### Scatter Plot")
                    other_columns = [col for col in df.columns if col != selected_column]
                    x_axis_column = st.selectbox("Select X-Axis Column", other_columns)

                    if x_axis_column:
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=df, x=x_axis_column, y=selected_column, ax=ax)
                        ax.set_title(f"Scatter Plot: {selected_column} vs {x_axis_column}")
                        st.pyplot(fig)

def handle_numeric_column(df, selected_column):
    return