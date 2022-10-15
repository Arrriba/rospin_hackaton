import streamlit as st
import leafmap.foliumap as leafmap

import io
from PIL import Image
import numpy as np
from geopy import Nominatim
from transformers import CLIPProcessor, CLIPModel

from selenium import webdriver
import os 
path = "./"
os.environ["PATH"] += os.pathsep + path + 'geckodriver.exe'

driver = webdriver.Firefox(path)
path = "./"

driver = webdriver.Firefox(path)

model = CLIPModel.from_pretrained("flax-community/clip-rsicd")
processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd")


address = st.text_input(label='Address, Town')
if address:
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode(address)

    m2 = leafmap.Map(minimap_control=True, draw_export=True, png_enabled=True, center=[location.latitude, location.longitude], zoom=18)
    m2.add_basemap("HYBRID")

    img_data = m2._to_png(1)
    img = Image.open(io.BytesIO(img_data))

    image = np.array(img)
    im = image[:600, 200:1100, :]
    img = Image.fromarray(im)

    img.save('md18.png')
    st.image(img)

    if img is not None:
        labels = ["residential area", "playground", "lake", "stadium", "forest", "airport", "park", "pool", "commercial"]
        inputs = processor(text=[f"a photo of a {l}" for l in labels], images=img, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image

        probs = logits_per_image.softmax(dim=1)

        p_idx = probs[0].argmax(dim=-1)
        prob = probs[0][p_idx]
        lbl = labels[p_idx]
        v = ['a', 'e', 'i', 'o', 'u']
        an = any([voc == lbl[0] for voc in v])
        lbl_str = f"{'an' if an else 'a'} {lbl}"
        st.write(f'We are {prob * 100:.2f}% confident that at this address there is {lbl_str}.')
