"""  
    # 1. Navigate to the project folder
    cd "e:\Data Quality\DataQualityProject"

    # 2. Create a virtual environment in the project folder
    python -m venv venv

    # 3. Activate the virtual environment
    # On Windows:
    venv\Scripts\activate

    # On macOS/Linux:
    # source venv/bin/activate

    # 4. Install the required packages
    pip install streamlit pandas ollama

    # 5. Verify the installed packages
    pip list

    # 6. Run the Streamlit app
    streamlit run RAG.py
"""

import streamlit as st
import pandas as pd
import ollama
import io
# App Title
st.title("Data Quality Task")
st.markdown("### الهدف هو بناء برنامج يساعد المستخدم في أن يعمل على الداتا بتاعته")

# Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("Error reading file: " + str(e))
else:
    st.warning("Please upload a dataset!")

# Sidebar for navigation
with st.sidebar:
    st.subheader("Navigation")
    menu = st.radio("Choose a task:", [
        "1. Dataset Info",
        "2. Describe Dataset",
        "3. Handle Missing Values",
        "4. Handle Duplicates",
        "5. Handle Outliers",
        "6. Chat using RAG"
    ])

# Dataset Info
def display_dataset_info():
    st.markdown("### Dataset Information")
    buffer = io.StringIO()  # Create a StringIO buffer
    df.info(buf=buffer)  # This fills the buffer with dataset info
    info_string = buffer.getvalue()  # Get the string content from the buffer
    st.text(info_string)

# Handle Missing Values
def handle_missing_values():
    st.subheader("Handle Missing Values")
    st.write("Missing Value Summary Before:")
    st.write(df.isnull().sum())

    columns_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
    if not columns_with_missing:
        st.success("No missing values found!")
        return

    col = st.selectbox("Select column to handle:", columns_with_missing)
    action = st.radio("Action:", ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"])

    if st.button("Apply Action"):
        if action == "Fill with Mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif action == "Fill with Median":
            df[col].fillna(df[col].median(), inplace=True)
        elif action == "Fill with Mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif action == "Drop Rows":
            df.dropna(subset=[col], inplace=True)
        st.success(f"Applied '{action}' to column '{col}'.")
        st.write("Missing Value Summary After:")
        st.write(df.isnull().sum())

# Handle Duplicates
def handle_duplicates():
    st.subheader("Handle Duplicates")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0 and st.button("Remove Duplicates"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicates removed!")
        st.write(f"Number of duplicate rows after removal: {df.duplicated().sum()}")

# Handle Outliers
def handle_outliers():
    st.subheader("Handle Outliers")
    if st.button("Remove Outliers using IQR"):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.success("Outliers removed using IQR!")
        st.write(df_outliers_removed.describe())
        st.dataframe(df_outliers_removed)

# Chat with RAG
def chat_with_rag():
    st.subheader("Chat with Dataset using RAG")

    def ollama_generate(query: str, model: str = "llama3.2:latest") -> str:
        """Generate a response using Ollama."""
        try:
            result = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
            return result.get("message", {}).get("content", "No response content.")
        except Exception as e:
            return f"Error: {e}"

    # Function to chat with CSV using Ollama
    def chat_with_csv_ollama(df, prompt, model="llama3.2:latest", max_rows=10):
        """Chat with a CSV using Ollama."""
        # Summarize dataset: Include column names, row count, and sample rows
        summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        column_info = "Columns:\n" + "\n".join([f"- {col} (type: {str(df[col].dtype)})" for col in df.columns])
        sample_data = f"Sample rows:\n{df.head(5).to_string(index=False)}"

        # Include data content (limit rows if necessary)
        data_content = f"The dataset:\n{df.head(max_rows).to_string(index=False)}"

        # Create the query
        query = f"""
        You are a data assistant. Here is the summary of the dataset:
        {summary}
        {column_info}
        {sample_data}

        {data_content}

        Based on this dataset, answer the following question:
        {prompt}
        """
        
        # Use the `ollama_generate` function to get the response
        return ollama_generate(query, model=model)

    # Initialize session state for query and response history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Stores the history as a list of dictionaries with roles and messages

    # App title
    st.title("ChatCSV powered by Ollama")

    # Upload CSV section
        

    if df is not None:
        # Read and display the CSV
        st.info("CSV Uploaded Successfully")
        
        st.dataframe(df, use_container_width=True)

        # Chat interface
        st.info("Chat Below")
        user_input = st.chat_input("Ask a question:")

        if user_input:
            # Add user query to the conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Generate response from Ollama
            with st.spinner("Generating response..."):
                assistant_response = chat_with_csv_ollama(df, user_input)

            # Add assistant response to the conversation
            st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

        # Display the conversation
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            elif message["role"] == "assistant":
                # Check if the message contains code blocks
                if "```" in message["content"]:
                    # Split by code blocks
                    code_blocks = message["content"].split("```")
                    for i, block in enumerate(code_blocks):
                        if i % 2 == 1:  # Odd indices are code blocks
                            st.code(block.strip(), language="python")  # Render as code
                        else:
                            if block.strip():  # Avoid rendering empty text
                                st.chat_message("assistant").markdown(block.strip())
                else:
                    st.chat_message("assistant").markdown(message["content"])

# Task Navigation
if df is not None:
    if menu == "1. Dataset Info":
        display_dataset_info()
    elif menu == "2. Describe Dataset":
        st.subheader("Dataset Description")
        st.write(df.describe())
    elif menu == "3. Handle Missing Values":
        handle_missing_values()
    elif menu == "4. Handle Duplicates":
        handle_duplicates()
    elif menu == "5. Handle Outliers":
        handle_outliers()
    elif menu == "6. Chat using RAG":
        chat_with_rag()
    


else:
    st.info("Upload a dataset to begin.")

# Download modified dataset
if df is not None:
    st.sidebar.download_button(
        label="Download Modified Dataset",
        data=df.to_csv(index=False),
        file_name="modified_dataset.csv",
        mime="text/csv"
    )





