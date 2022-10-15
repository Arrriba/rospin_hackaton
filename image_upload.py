import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import io

st.title('ROSPIN Hackathon ')

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("flax-community/clip-rsicd")
processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd")


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    print(type(uploaded_file))
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data))
    image = np.asarray(image)
    st.image(image)
    labels = ["residential area", "playground","lake", "stadium", "forest", "airport", "park", "pool"]
    inputs = processor(text=[f"a photo of a {l}" for l in labels], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs.shape)
    # probs = probs[:,probs.argmax(dim=-1)]
    for l, p in zip(labels, probs[0]):
        st.write(f"{l:<16} {p*100:.2f}%")
        # st.write(probs.argmax(dim=-1))





