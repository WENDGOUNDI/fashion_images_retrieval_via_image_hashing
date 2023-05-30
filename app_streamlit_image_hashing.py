import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os
import pickle
from PIL import Image
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import base64
import pandas as pd
import imagehash
import glob

with open("logo.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="data:image/png;base64,{data}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("ECOMMERCE PRODUCT SIMILARITY SEARCH WEBAPP")



# LOAD IMAGE
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["png","jpg", "jpeg"])    


if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    img = Image.open(BytesIO(bytes_data))
    img = img.convert('RGB')
    display_img = img.resize((150,150))
    st.image(display_img)


#df = pd.read_csv("my_images_DB.csv")
#images_list = df["images_list"]
#hash_images_list = df["hash_images_list"]

dataset_path = "C:/Users/savadogo_abdoul/Desktop/Learn/image_search_similarities/data_subset/"

images_list = list()
hash_images_list = list()

for image_file in glob.glob(dataset_path+"*.jpg"):
    cal_hash =  imagehash.average_hash(Image.open(image_file))
    images_list.append(image_file)
    hash_images_list.append(cal_hash)




if st.button('Search'):
    # Calculate hash value between target image and our database

    target_image = display_img
    hash_target_image =  imagehash.average_hash(target_image)

    result_list = list()

    for i in hash_images_list:
        result_list.append( i - hash_target_image)


    index_list = []
    hash_values = []

    for index , i in enumerate(result_list):
        #print(index,i)
        index_list.append(index)
        hash_values.append(i)

    # Create a dictionary  to save hash values and their indexes
    mydict = dict(zip(index_list, hash_values) )

    # Sort above dictionary from lowest to highest 
    sorted_dict = dict(sorted(mydict.items(), key=lambda x:x[1]))

    # Extract indexes to retreive similar images
    dictkeys = list(sorted_dict.keys())

    # Retrieve our Top K similar images
    top_k_similar = dictkeys[:7]



    # Load the images
    img0 = plt.imread(images_list[top_k_similar[0]])
    img1 = plt.imread(images_list[top_k_similar[1]])
    img2 = plt.imread(images_list[top_k_similar[2]])
    img3 = plt.imread(images_list[top_k_similar[3]])
    img4 = plt.imread(images_list[top_k_similar[4]])
    img5 = plt.imread(images_list[top_k_similar[5]])

    # Display the images in a grid
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(10,2))
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    axes[2].imshow(img2)
    axes[3].imshow(img3)
    axes[4].imshow(img4)
    axes[5].imshow(img5)
    plt.show()


    st.header(" Displayed Similar Images to the Input Image Randomly.")
    st.image([img1, img2, img3, img4, img5], width=125)
    