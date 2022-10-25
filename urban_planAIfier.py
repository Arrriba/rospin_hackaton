import streamlit as st
import leafmap.foliumap as leafmap
from transformers import CLIPProcessor, CLIPModel

import io
import torch
from PIL import Image
import numpy as np
import cv2
from geopy import Nominatim
import albumentations as A
from gan import GeneratorUNet
import glob
from typing import Union

from streamlit_image_comparison import image_comparison


def cosine_similarity(x1: Union[np.ndarray, torch.tensor], x2: Union[np.ndarray, torch.tensor]):

  x1 = x1 / x1.norm()
  x2 = x2 / x2.norm()

  sim = (x1 @ x2.T).detach().numpy()[0]#.item()

  return sim

def compare_images(img1, img2):
    img1, img2 = np.array(img1), np.array(img2)

    mask = (img1 != img2).mean(axis=-1)
    mask = np.where(mask==0, 0, 1).astype('bool')

    img1[mask] = (255, 0, 0)

    return img1


# st.set_page_config(layout="wide")
st.header("Urban PlanAIfier")
st.subheader("A new AI analytics perspective for your city.")
col1, col2 = st.columns((3, 3))


dataset_files = glob.glob(r'./images/*.*')
gan_model = GeneratorUNet()
gan_path = r'./weights/gen.tar.pth'
gan_model.load_state_dict(torch.load(gan_path))
gan_model.eval()

model = CLIPModel.from_pretrained("flax-community/clip-rsicd")
processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd")

test_transforms = A.Compose([A.Resize(256, 256),
                             A.Normalize((0, 0, 0), (1., 1., 1.))])

labels = ["residential area", "playground", "lake", "stadium", "forest", "airport", "park", "pool", "commercial center"]

# with col1:
address = st.text_input(label='Address, Town')
if address:
    locator = Nominatim(user_agent='myGeocoder')
    location = locator.geocode(address)

    m2 = leafmap.Map(minimap_control=True, draw_export=True, png_enabled=True, center=[location.latitude, location.longitude], zoom=18)
    m2.add_basemap("HYBRID")

    img_data = m2._to_png(1)
    img = Image.open(io.BytesIO(img_data))

    image = np.array(img)
    # print(image.shape)
    im = image[:600, 200:1100, :]
    im1 = im
    img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (900, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        pred_img = test_transforms(image=im1[:, :, :3])['image']
        pred_img = torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0)
        pred = gan_model(pred_img) * 255.
        pred = pred.squeeze(0).permute(1, 2, 0).detach().numpy().astype('uint8')
        pred = cv2.resize(pred, (900, 600))

    #img.save('md.png')
    # st.image(img)
    image_comparison(img1=img, img2=pred, label1="satellite image", label2="map image", width=700,
starting_position=50,
show_labels=True)


    if img is not None:
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

# with col2:
uploaded_img = st.file_uploader(label='Enter the image from the present to compare with the past', type=['jpg', 'png', 'jpeg'])
if uploaded_img is not None:
    bytes_data = uploaded_img.getvalue()
    pil_uploaded_image = Image.open(io.BytesIO(bytes_data))
    image = np.asarray(pil_uploaded_image)

    dset_images = [np.array(Image.open(fname)) for fname in dataset_files]

    inputs = processor(text=[f"a photo of a {l}" for l in labels], images=[image] + dset_images, return_tensors="pt", padding=True)
    out = model(**inputs)
    embedd = out.image_embeds

    x1, x2 = embedd.chunk(2, dim=0)
    sim = cosine_similarity(x1, x2)

    max_sim = np.argmax(sim, -1)
    selected_image = Image.open(dataset_files[max_sim])

    image_comparison(img1=pil_uploaded_image, img2=selected_image, label1="present",
label2="past",
width=700,
starting_position=50,
show_labels=True)

    button = st.button('Spot differences')

    if button:
        img = compare_images(pil_uploaded_image, selected_image)
        st.image(img, width=700)


        