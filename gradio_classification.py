# -*- coding: utf-8 -*-
"""gradio_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y5mRBUPfukeouW9iXanglG_yL-isgXbG
"""

# Install necessary libraries in Google Colab
!pip install gradio transformers torch safetensors

# Importing libraries
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import torch
from safetensors.torch import load_file

# Define the paths to your model and tokenizer files
model_directory = "/model1.zip"  # Path to the directory containing model files
tokenizer_directory = "/tokenizer1.zip"  # Path to the directory containing tokenizer files

# Assuming your tokenizer was saved using save_pretrained
tokenizer_path = tokenizer_directory # Update with the correct path if needed
# Load the custom DistilBERT tokenizer from the specified directory
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

# Assuming your model was saved using .save_pretrained, you likely have a config.json in the model_directory
config_path = f"{model_directory}/config.json" # Update with the correct path if needed
# Load the configuration
config = DistilBertConfig.from_pretrained(config_path)

# Initialize the model with the configuration
model = DistilBertForSequenceClassification(config)

# Load the state dict from the safetensors file
state_dict = load_file(f"{model_directory}/model.safetensors") # Update with the correct filename if needed
# Load the state dict into the model
model.load_state_dict(state_dict)

# Function to classify an article as "Risk" or "No Risk"
def classify_article(article):
    inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    # Assuming label 0 is "No Risk" and label 1 is "Risk"
    return "Risk" if prediction == 1 else "No Risk"

# Gradio interface setup
with gr.Blocks() as risk_classifier:
    gr.Markdown("# Article Risk Categorization")

    # Text input for a single news article
    article_input = gr.Textbox(label="Enter News Article", lines=5, placeholder="Paste the news article here...")

    # Output: Categorization result
    category_output = gr.Textbox(label="Category (Risk/No Risk)")

    # Run classification
    analyze_button = gr.Button("Classify Article")
    analyze_button.click(fn=classify_article, inputs=article_input, outputs=category_output)

# Launch the Gradio app in Colab
risk_classifier.launch(share=True)

