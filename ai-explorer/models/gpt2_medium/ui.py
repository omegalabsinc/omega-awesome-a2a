import streamlit as st
from model import GPT2MediumModel
import time

@st.cache_resource
def load_model():
    return GPT2MediumModel()

st.title("GPT2-Medium Explorer")
st.write("A lightweight but capable text generation model")

model = load_model()

prompt = st.text_area("Enter your prompt:", height=100)
max_length = st.slider("Maximum length:", 50, 200, 100)
temperature = st.slider("Temperature:", 0.1, 1.0, 0.7)

if st.button("Generate"):
    with st.spinner("Generating..."):
        start_time = time.time()
        output = model.generate(prompt, max_length=max_length, temperature=temperature)
        gen_time = time.time() - start_time
        st.write("Generated Text:")
        st.write(output)
        st.info(f"Generation time: {gen_time:.2f} seconds")
