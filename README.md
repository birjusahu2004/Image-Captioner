# AI Image Captioning System

An end-to-end Image Captioning system built using Deep Learning (CNN + LSTM) and trained on the Flickr8k dataset.

## 🚀 Live Demo
https://huggingface.co/spaces/birjusahu2004/Image-Captioner

## 🛠️ Tech Stack
- **Computer Vision:** DenseNet201 (Feature Extraction)
- **NLP:** LSTM Networks
- **Framework:** TensorFlow / Keras
- **Deployment:** Gradio & Hugging Face Spaces

## 📖 Project Structure
- `image_captioning.ipynb`: Full training pipeline (Data augmentation, Feature extraction, Training).
- `app.py`: Inference script for the Gradio web interface.
- `requirements.txt`: Necessary Python dependencies.

## 📝 How it Works
1. **Encoder:** A pre-trained DenseNet201 model extracts a 1920-dimensional feature vector from the input image.
2. **Decoder:** An LSTM-based recurrent neural network takes the image features and generates a descriptive sequence of words (caption).
3. **Inference:** The model uses greedy search to predict the next word in the sequence until an `endseq` token is reached.
