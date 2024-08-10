# import streamlit as st
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# import numpy as np
# import io
# from pandasai.llm.google_gemini import GoogleGemini
# from pandasai import SmartDataframe

# # Function to make API request to Gemini
# def gemini_request(prompt, api_key):
#     url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
#     headers = {'Content-Type': 'application/json'}
#     data = {'contents': [{'parts': [{'text': prompt}]}]}
#     response = requests.post(url, headers=headers, json=data, params={'key': api_key})
#     return response

# def extract_chart_info(recommendation):
#     # Extract chart type
#     chart_type_match = re.search(r"Create a (\w+(?:\s+\w+)*) (?:chart|plot|graph)", recommendation, re.IGNORECASE)
#     chart_type = chart_type_match.group(1).strip().lower() if chart_type_match else "bar"

#     # Extract axis labels and title
#     x_label = re.search(r"Label the x-axis [\"'](.+?)[\"']", recommendation)
#     y_label = re.search(r"Label the y-axis [\"'](.+?)[\"']", recommendation)
#     title = re.search(r"Title the chart [\"'](.+?)[\"']", recommendation)

#     return {
#         'chart_type': chart_type,
#         'x_label': x_label.group(1) if x_label else None,
#         'y_label': y_label.group(1) if y_label else None,
#         'title': title.group(1) if title else None
#     }

# def create_chart(df, chart_info):
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # Ensure df is a DataFrame
#     if isinstance(df, pd.Series):
#         df = df.to_frame()

#     # Reset index if it's not a range index
#     if not isinstance(df.index, pd.RangeIndex):
#         df = df.reset_index()

#     # Determine x and y based on DataFrame structure
#     if len(df.columns) == 2:
#         x, y = df.columns
#     else:
#         x = df.columns[0]
#         y = df.columns[1] if len(df.columns) > 1 else df.columns[0]

#     # Create the appropriate chart
#     if chart_info['chart_type'] in ['bar', 'bar chart']:
#         sns.barplot(x=x, y=y, data=df, ax=ax)
#     elif chart_info['chart_type'] in ['line', 'line chart']:
#         sns.lineplot(x=x, y=y, data=df, ax=ax)
#     elif chart_info['chart_type'] in ['scatter', 'scatter plot']:
#         sns.scatterplot(x=x, y=y, data=df, ax=ax)
#     elif chart_info['chart_type'] in ['pie', 'pie chart']:
#         df.plot(kind='pie', y=y, labels=df[x], ax=ax, autopct='%1.1f%%')
#     else:
#         st.error(f"Unsupported chart type: {chart_info['chart_type']}")
#         return None

#     # Set labels and title
#     ax.set_xlabel(chart_info['x_label'] or x)
#     ax.set_ylabel(chart_info['y_label'] or y)
#     ax.set_title(chart_info['title'] or f"{chart_info['chart_type'].capitalize()} of {y} by {x}")

#     # Rotate x-axis labels if it's not a pie chart
#     if chart_info['chart_type'] not in ['pie', 'pie chart']:
#         plt.xticks(rotation=45, ha='right')

#     plt.tight_layout()
#     return fig

# def determine_axes(df):
#     if isinstance(df, pd.Series):
#         return df.index.name or "Index", df.name or "Value"
#     elif len(df.columns) == 2:
#         return df.columns[0], df.columns[1]
#     else:
#         numeric_columns = df.select_dtypes(include=[np.number]).columns
#         if len(numeric_columns) > 0:
#             return df.columns[0], numeric_columns[0]
#         else:
#             return df.columns[0], df.columns[1]



# def get_viz_recommendation(filtered_df, api_key):
#     numerical_columns, categorical_columns = get_numerical_categorical_columns(filtered_df)
    
#     visualization_prompt = f"""
#     Based on the following DataFrame, generate a visualization:
#     DataFrame:
#     {filtered_df.head().to_string()}
#     The DataFrame contains {numerical_columns} as numerical columns and {categorical_columns} as categorical columns.
#     Generate a prompt for creating the most suitable chart type. Include instructions for labeling the axes, titling the chart.
#     Make it short and concise and don't provide any code.
#     """
    
#     headers = {'Content-Type': 'application/json'}
#     data = {'contents': [{'parts': [{'text': visualization_prompt}]}]}
#     url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    
#     response = requests.post(url, headers=headers, json=data, params={'key': api_key})
    
#     if response.status_code == 200:
#         return response.json()['candidates'][0]['content']['parts'][0]['text']
#     else:
#         return None

# def get_numerical_categorical_columns(df):
#     numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#     return numerical_columns, categorical_columns
    
# # Streamlit app
# st.title("Data Analysis and Visualization App")

# # File uploader
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Read the CSV file
#     df = pd.read_csv(uploaded_file)
#     df =df.drop(columns=['Unnamed: 0'])
#     st.write("Dataset Preview:")
#     st.dataframe(df.head())

#     # Prompt input
#     prompt = st.text_area("Enter your filtration prompt:")
    
#     # API key input (you might want to use st.secrets for this in production)
#     api_key = st.secrets["gemini_api"]["gemini_API"]
#     pandas_api_key = st.secrets["pandasai_api"]["pandas_api"]

# if st.button("Submit"):
#     if not api_key:
#         st.error("Please enter your Gemini API key.")
#     elif not prompt:
#         st.error("Please enter a filtration prompt.")
#     else:
#         full_prompt = f"""
#         You are an expert data analyst. Given the following database in the form of a DataFrame, write a Pandas statement to fulfill the following request:

#         Database: {df.head().to_string()}

#         Request: {prompt}

#         Please provide:
#         1. The Pandas query to fulfill the request, stored in a variable called filtered_df.
#         2. A brief explanation of the query and the resulting DataFrame.

#         Format your response as follows:
#         ```python
#         filtered_df = ...
#         ```
#         Explanation: ...
#         """

#         # Make API request for filtration
#         response = gemini_request(full_prompt, api_key)

#         if response.status_code == 200:
#             full_response = response.json()['candidates'][0]['content']['parts'][0]['text']
            
#             # Split the response into code and explanation
#             parts = full_response.split("```")
#             if len(parts) >= 3:
#                 code_part = parts[1].strip()
#                 explanation_part = parts[2].strip()
                
#                 # Extract the code
#                 filtered_code = code_part.replace("python\n", "").strip()
                
#                 # Extract the explanation
#                 explanation = explanation_part.split("Explanation:", 1)[-1].strip()
                
#                 st.subheader("Pandas Query:")
#                 st.code(filtered_code, language='python')
                
#                 try:
#                     # Execute the filtered DataFrame code
#                     exec(filtered_code)
#                     if isinstance(filtered_df, pd.Series):
#                         filtered_df = filtered_df.to_frame()
#                     elif not isinstance(filtered_df, pd.DataFrame):
#                         if isinstance(filtered_df, (list, dict)):
#                             filtered_df = pd.DataFrame(filtered_df)
#                         else:
#                             filtered_df = pd.DataFrame([filtered_df])
#                     st.subheader("Resulting DataFrame:")
#                     st.dataframe(filtered_df)
                    
#                     st.subheader("Explanation:")
#                     st.write(explanation)
                    
#                     # Proceed with visualization
#                     viz_recommendation = get_viz_recommendation(filtered_df, api_key)
                    
#                     if viz_recommendation:
#                         st.subheader("Visualization Recommendation:")
#                         st.write(viz_recommendation)

#                         try:
#                             # Ensure filtered_df is a Series or DataFrame
#                             if not isinstance(filtered_df, (pd.Series, pd.DataFrame)):
#                                 filtered_df = pd.Series(filtered_df)
                            
#                             # Create the chart using the modified create_chart function
#                             fig = create_chart(filtered_df, viz_recommendation)
                            
#                             if fig:
#                                 # Display the plot using Streamlit
#                                 st.pyplot(fig)
#                             else:
#                                 st.error("Failed to create the chart.")
#                         except Exception as e:
#                             st.error(f"Error generating visualization: {str(e)}")
#                             st.error(f"DataFrame type: {type(filtered_df)}")
#                             st.error(f"DataFrame content: {filtered_df}")
#                     else:
#                         st.error("Failed to get visualization recommendation.")
#                 except Exception as e:
#                     st.error(f"Error executing the query: {str(e)}")
#             else:
#                 st.error("Unexpected response format from the API.")
#         else:
#             st.error(f"API request failed: {response.status_code} - {response.text}")



import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from pandasai.llm.google_gemini import GoogleGemini
from pandasai import SmartDataframe

# Function to make API request to Gemini
def gemini_request(prompt, api_key):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
    headers = {'Content-Type': 'application/json'}
    data = {'contents': [{'parts': [{'text': prompt}]}]}
    response = requests.post(url, headers=headers, json=data, params={'key': api_key})
    return response

def get_numerical_categorical_columns(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_columns, categorical_columns

# Streamlit app
st.title("Data Analysis and Visualization App")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
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
            prompt_template = """
            You are an expert data analyst. Given the following database in the form of a DataFrame, write a {query_type} statement to fulfill the following request:
            Database: {database}
            Request: {request}
            Please provide:
            1. The Pandas query to fulfill the request, stored in a variable called filtered_df.
            2. A brief explanation of the query and the resulting DataFrame.
            Format your response as follows:
            ```python
            filtered_df = ...
            ```
            explanation: ...
            """
            filled_prompt = prompt_template.format(query_type="Pandas", database=df.head().to_string(), request=prompt)

            # Make API request for filtration
            response = gemini_request(filled_prompt, api_key)

            if response.status_code == 200:
                full_response = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # Split the response into code and explanation
                parts = full_response.split("```")
                if len(parts) >= 3:
                    code_part = parts[1].strip()
                    explanation_part = parts[2].strip()
                    
                    # Extract the code
                    filtered_code = code_part.replace("python\n", "").strip()
                    
                    # Extract the explanation
                    explanation = explanation_part.split("explanation:", 1)[-1].strip()
                    
                    st.subheader("Pandas Query:")
                    st.code(filtered_code, language='python')
                    
                    # Execute the filtered DataFrame code
                    exec(filtered_code)
                    
                    filtered_df = filtered_df.reset_index()
                    
                    st.subheader("Resulting DataFrame:")
                    st.dataframe(filtered_df)
                    
                    st.subheader("Explanation:")
                    st.write(explanation)
                    
                    # Get numerical and categorical columns
                    numerical_columns, categorical_columns = get_numerical_categorical_columns(filtered_df)
                    
                    # Create visualization prompt
                    visualization_prompt = """
                    Based on the following DataFrame, generate a visualization:
                    DataFrame:
                    {dataframe}
                    The DataFrame contains {numerical_columns} as numerical columns and {categorical_columns} as categorical columns.
                    Generate a prompt for creating the most suitable chart type. Include instructions for labeling the axes, titling the chart.
                    Make it short and concise and dont provide any code.
                    """
                    visualization_prompt_filled = visualization_prompt.format(
                        dataframe=filtered_df.head().to_string(),
                        numerical_columns=numerical_columns,
                        categorical_columns=categorical_columns
                    )
                    
                    # Get visualization recommendation
                    viz_response = gemini_request(visualization_prompt_filled, api_key)
                    
                    if viz_response.status_code == 200:
                        visualization_recommendation = viz_response.json()['candidates'][0]['content']['parts'][0]['text']
                        st.subheader("Visualization Recommendation:")
                        st.write(visualization_recommendation)
                        
                        # Use PandasAI to generate the visualization
                        gemini_llm = GoogleGemini(api_key=api_key)
                        smart_filter_df = SmartDataframe(filtered_df, config={"llm": gemini_llm})
                        
                        try:
                            fig = smart_filter_df.chat(visualization_recommendation)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                    else:
                        st.error("Failed to get visualization recommendation.")
                else:
                    st.error("Unexpected response format from the API.")
            else:
                st.error(f"API request failed: {response.status_code} - {response.text}")