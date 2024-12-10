import streamlit as st  
from explorer.mistral_handler import MistralHandler  

# Set up Mistral API key  
api_key = st.secrets["MISTRAL_API_KEY"]  
handler = MistralHandler(api_key)  

st.title("AI Explorer with Mistral Integration")  
model = st.selectbox("Choose a model:", ["OpenAI", "Mistral"])  
prompt = st.text_area("Enter your prompt:")  

if st.button("Generate Response"):  
    with st.spinner("Generating response..."):  
        try:  
            response = handler.generate_response(prompt) if model == "Mistral" else "OpenAI response placeholder"  
            st.success(response)  
        except Exception as e:  
            st.error(f"Error: {e}")  
