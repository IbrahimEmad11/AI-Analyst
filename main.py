import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import io


# Function to make API request to Gemini
def gemini_request(prompt, api_key):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    headers = {'Content-Type': 'application/json'}
    data = {'contents': [{'parts': [{'text': prompt}]}]}
    response = requests.post(url, headers=headers, json=data, params={'key': api_key})
    return response

def extract_chart_info(recommendation):
    # Extract chart type
    chart_type_match = re.search(r"Recommended Visualization: (.+)", recommendation)
    chart_type = chart_type_match.group(1).strip().lower() if chart_type_match else "bar"

    # Extract customization recommendations
    customizations = []
    if "sorting" in recommendation.lower():
        customizations.append("sort")
    if "label" in recommendation.lower():
        customizations.append("labels")
    if "color" in recommendation.lower():
        customizations.append("color")

    return chart_type, customizations

def determine_axes(df):
    if isinstance(df, pd.Series):
        return df.index.name or "Index", df.name or "Value"
    elif len(df.columns) == 2:
        return df.columns[0], df.columns[1]
    else:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            return df.columns[0], numeric_columns[0]
        else:
            return df.columns[0], df.columns[1]

def create_chart(df, recommendation):
    chart_type, customizations = extract_chart_info(recommendation)
    x_axis, y_axis = determine_axes(df)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set a color palette if requested
    if "color" in customizations:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)

    if chart_type == "bar chart":
        if isinstance(df, pd.Series):
            if "sort" in customizations:
                df = df.sort_values(ascending=False)
            df.plot(kind='bar', ax=ax)
           
            if "labels" in customizations:
                for i, v in enumerate(df):
                    ax.text(i, v, f'{v:,}', ha='center', va='bottom')
        else:
            if "sort" in customizations:
                df = df.sort_values(y_axis, ascending=False)
            sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
           
            if "labels" in customizations:
                for i, v in enumerate(df[y_axis]):
                    ax.text(i, v, f'{v:,}', ha='center', va='bottom')
    elif chart_type == "line chart":
        sns.lineplot(data=df, ax=ax)
    elif chart_type == "scatter plot":
        if isinstance(df, pd.Series):
            ax.scatter(range(len(df)), df.values)
        else:
            sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
    elif chart_type == "histogram":
        if isinstance(df, pd.Series):
            sns.histplot(df, kde=True, ax=ax)
        else:
            sns.histplot(df[x_axis], kde=True, ax=ax)
    elif chart_type == "pie chart":
        if isinstance(df, pd.Series):
            df.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        else:
            df.plot(kind='pie', y=y_axis, autopct='%1.1f%%', ax=ax)
    else:
        print(f"Unsupported chart type: {chart_type}")
        return None

    ax.set_title(f'{chart_type.capitalize()} of {y_axis} by {x_axis}')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig



# Update the visualization prompt to ask for code
def get_viz_recommendation(filtered_df, api_key):
    viz_prompt = f"""
    Based on the following DataFrame, recommend a suitable visualization type and provide the Python code to create it:
    
    DataFrame:
    {filtered_df.to_string()}
    
    Please provide:
    1. A brief description of the recommended visualization type and how it helps in understanding the data.
    2. The complete Python code to create this visualization using matplotlib or seaborn.
    
    Ensure the code is complete and can be executed directly. Use the variable name 'filtered_df' for the DataFrame in your code.
    """
    
    viz_response = gemini_request(viz_prompt, api_key)
    
    if viz_response.status_code == 200:
        return viz_response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return None
    
    
# Streamlit app
st.title("Data Analysis and Visualization App")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    df =df.drop(columns=['Unnamed: 0'])
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Prompt input
    prompt = st.text_area("Enter your filtration prompt:")
    
    # API key input (you might want to use st.secrets for this in production)
    api_key = st.secrets["gemini_api"]["gemini_API"]

    if st.button("Submit"):
        if not api_key:
            st.error("Please enter your Gemini API key.")
        elif not prompt:
            st.error("Please enter a filtration prompt.")
        else:
            # Create the full prompt
            full_prompt = f"""
            You are an expert data analyst. Given the following database in the form of a DataFrame, write a Pandas statement to fulfill the following request:
            Database: {df.head().to_string()}
            Request: {prompt}
            Make the answer short and concise and store in variable called filtered_df .
            """

            # Make API request for filtration
            response = gemini_request(full_prompt, api_key)

            if response.status_code == 200:
                filtered_code = response.json()['candidates'][0]['content']['parts'][0]['text'].strip("```python\n").strip("```")
                
                # Execute the filtered DataFrame code
                exec(filtered_code)
                
                # Display the filtered DataFrame
                st.write("Filtered DataFrame:")
                st.dataframe(filtered_df)

                # Create visualization prompt
                viz_prompt = f"""
                Based on the following DataFrame, recommend a suitable visualization type for it:
                DataFrame:
                {filtered_df.head().to_string()}
                The DataFrame contains:
                {filtered_df.describe(include='all').to_string()}
                Provide a brief description of the recommended visualization type and how it helps in understanding the data.
                """

                # Get visualization recommendation and code
                viz_recommendation = get_viz_recommendation(filtered_df, api_key)
                
                if viz_recommendation:
                    st.write("Visualization Recommendation:")
                    st.write(viz_recommendation)
                    
                    # Extract the code from the recommendation
                    code_start = viz_recommendation.find("```python")
                    code_end = viz_recommendation.find("```", code_start + 1)
                    if code_start != -1 and code_end != -1:
                        viz_code = viz_recommendation[code_start+9:code_end].strip()
                        
                        st.write("Visualization Code:")
                        st.code(viz_code, language='python')
                        
                        # Execute the code
                        try:
                            # Capture the plot output
                            buf = io.BytesIO()
                            exec(viz_code)
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            
                            # Display the plot
                            st.image(buf)
                        except Exception as e:
                            st.error(f"Error executing visualization code: {str(e)}")
                    else:
                        st.error("No Python code found in the visualization recommendation.")
                else:
                    st.error("Failed to get visualization recommendation.")