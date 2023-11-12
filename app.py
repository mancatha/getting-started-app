from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
import gradio as gr

# Load the model and tokenizer
# setup
model_name = "benmanks/sentiment_analysis_trainer_model"  
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("benmanks/sentiment_analysis_trainer_model")

# Function
def preprocess(text):
    # Preprocess text (username and link placeholders)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_analysis(text):
    text = preprocess(text)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class probabilities
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]

    # Map the scores to labels
    labels = ['Negative', 'Neutral', 'Positive']
    scores_dict = {label: score for label, score in zip(labels, scores)}

    return scores_dict

title = "Sentiment Analysis Application\n\n\nThis application assesses if a twitter post relating to vaccination "
description = "This application assesses if a twitter post relating to vaccination is positive,neutral or negative"

demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your tweet here..."),
    outputs=gr.Label(num_top_classes=3),
    examples=[["The Vaccine is harmful!"],
              ["I cant believe people don't vaccinate their kids"],
              ["FDA think just not worth the AE unfortunately"],
              ["For a vaccine given to healthy"],
              ["covid is hoax"]],
    title=title,
    description=description
    
)
if __name__ == " _main_":

    demo.launch(server_name = "0.0.0.0", server_port = 7860)