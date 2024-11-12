import streamlit as st
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import io

# Load the trained model (assuming model is saved locally or from Hugging Face Hub)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('/content/drive/MyDrive/model')  # Use the model directory
    model.eval()  # Set to evaluation mode
    return model

# Load the tokenizer from the saved directory (or Hugging Face Hub)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/tokenizer')  # Use the tokenizer directory
    return tokenizer

# Classification function using tokenizer
def classify_article(article_text, model, tokenizer):
    tokens = tokenizer(
        [article_text],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    prediction = output.logits.argmax(dim=1).item()
    categories = ['Risk', 'No Risk', 'Issue', 'Opportunity']
    return categories[prediction]

# Streamlit app interface
st.title('Article Classification App')

uploaded_file = st.file_uploader("Upload an article (text or .txt file)", type=["txt"])

if uploaded_file is not None:
    # Read the uploaded file
    article_text = uploaded_file.read().decode("utf-8")
    
    # Display article content
    st.write("Article Content:")
    st.text(article_text)
    
    # Load the model and tokenizer
    model = load_model()
    tokenizer = load_tokenizer()

    # Button to trigger classification
    if st.button("Classify Article"):
        prediction = classify_article(article_text, model, tokenizer)
        st.write(f"The article is classified as: **{prediction}**")
