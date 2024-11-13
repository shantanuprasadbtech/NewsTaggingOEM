%%writefile app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

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

    # Debugging output to see the raw prediction
    st.write(f"Raw prediction: {prediction}")
    
    # Define the mapping of classes to their desired outputs
    if prediction == 0:  # Assuming 0 corresponds to "Risk"
        return "Risk, Issue"
    elif prediction == 1:  # Assuming 1 corresponds to "No Risk"
        return "No Risk, Opportunity"
    else:
        return "Unknown Classification"

# Function to scrape article content from URL using BeautifulSoup
def scrape_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request fails
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Locate the div with class "post-content"
        content_div = soup.find('div', class_='post-content')
        
        # Extract text from all <p> tags within this div, ignoring 'meta' and 'excerpt' classes
        paragraphs = content_div.find_all('p') if content_div else []
        paragraph_texts = [p.get_text() for p in paragraphs if 'meta' not in p.get('class', []) and 'excerpt' not in p.get('class', [])]
        
        # Join all the paragraph texts into a single string
        paragraph_text = ' '.join(paragraph_texts)
        
        return paragraph_text or "No content found in the specified div."
    except Exception as e:
        return f"Error fetching article content: {e}"

# Streamlit app interface
st.title('Article Classification App')

# Select the input method
input_method = st.radio("Select Input Method:", ("Upload .txt file", "Paste article content", "Enter article URL"))

# Load the model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

if input_method == "Upload .txt file":
    uploaded_file = st.file_uploader("Upload an article (text or .txt file)", type=["txt"])

    if uploaded_file is not None:
        # Read the uploaded file
        article_text = uploaded_file.read().decode("utf-8")

        # Display article content
        st.write("Article Content:")
        st.text(article_text)

        # Button to trigger classification
        if st.button("Classify Article"):
            prediction = classify_article(article_text, model, tokenizer)
            st.write(f"The article is classified as: **{prediction}**")

elif input_method == "Paste article content":
    article_text_input = st.text_area("Enter the article content here:")

    if article_text_input:
        st.write("Entered Article Content:")
        st.write(article_text_input)

        # Button to trigger classification
        if st.button("Classify Article"):
            prediction = classify_article(article_text_input, model, tokenizer)
            st.write(f"The article is classified as: **{prediction}**")

elif input_method == "Enter article URL":
    url = st.text_input("Enter the URL of the article:")

    if url:
        article_text = scrape_article_content(url)
        if "Error" in article_text:
            st.error(article_text)
        else:
            st.write("Fetched Article Content:")
            st.write(article_text)

            # Button to trigger classification
            if st.button("Classify Article"):
                prediction = classify_article(article_text, model, tokenizer)
                st.write(f"The article is classified as: **{prediction}**")
