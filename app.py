# This is a template for your app.py file on Hugging Face
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. Load your saved assets
# Ensure these files are uploaded to the root of your HF Space
model = load_model('model.keras')
fe = load_model('feature_extractor.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34 # Based on Flickr8k preprocessing

def predict(input_img):
    # Preprocess
    img = input_img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Feature Extraction
    image_features = fe.predict(img_array, verbose=0)

    # Caption Generation
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat), None)
        if word is None or word == "endseq":
            break
        in_text += " " + word
    
    return in_text.replace("startseq", "").strip()

# 2. Launch Interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="AI Image Captioner",
    description="Deployment of my Flickr8k trained model."
)
iface.launch()