import streamlit as st
import os
# from segmentation.instance.instance_segmentation import do_instance
from semantic_segmentation import do_semantic
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

PAGE_TITLE = "Multi-class Semantic Segmentation of Medical Images (MR-Linacs on the Torso)"

LABEL_CONFIG_FN = r"label_config (1).json"
with open(LABEL_CONFIG_FN, 'r') as f:
    label_config = json.load(f)

N_LABELS = len(label_config)
LABEL_NAMES = sorted(label_config, key=lambda k: label_config[k]['index'])
LABEL_COLORS = np.array([label_config[n]['color'] for n in LABEL_NAMES])


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def file_selector_ui():
    # Select a file
    folder_path = '.'
    options = st.selectbox('get the file path',
                           ['Select a file in current directory', 'Change directory'])  # ,'upload from computer'])
    #     if options == 'upload from computer':
    #         # folder_path = st.file_uploader('Enter folder path', type='jpg')
    #         # folder_path = folder_path.read()
    #         #folder_path = st.text_input('Enter folder path', '.')
    #         #st.file_uploader("Upload a PNG image", type=([".png"]))
    #         # if file_png:
    #         #     file_png_bytes = st.file_reader(file_png)
    #         #     st.image(file_png_bytes)

    #         file_png = st.file_uploader("Upload a PNG image", type=([".png"]))

    #         if file_png:
    #             # file_png_bytes = st.file_reader(file_png)
    #             # st.image(file_png_bytes)
    #             # load_image_from_path(file_png)
    #             return file_png

    if options == 'Change directory':
        folder_path = st.text_input('Enter folder path', '.')
    print(folder_path)
    filename = file_selector(folder_path=folder_path)
    st.write('You selected `%s`' % filename)

    return filename


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    image_path = file_selector_ui()
    image_path = os.path.abspath(image_path)

    if os.path.isfile(image_path) is True:
        img = mpimg.imread(image_path)
        plt.figure(figsize=(3, 3))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        st.pyplot(plt)
        file_name = os.path.basename(image_path)
        _, file_extension = os.path.splitext(image_path)
        if file_extension == ".png":
            if st.button("Segmentation"):
                start_time_2 = time.time()
                semantic_image, img = do_semantic(image_path)
                mask = LABEL_COLORS[semantic_image]
                end_time_2 = time.time() - start_time_2
                plt.figure(figsize=(1, 1))
                plt.axis('off')
                plt.imshow(img)
                plt.imshow(mask, alpha=0.5)
                plt.show()
                st.pyplot(plt)
                # st.write(end_time_2)


if __name__ == "__main__":
    main()
