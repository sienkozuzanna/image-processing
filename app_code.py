import streamlit as st
from algorithms import *
from algorithms2 import *
from PIL import Image
import numpy as np
import io

st.set_page_config(layout="wide")

st.title('Image processing app')

with st.container():
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

with st.sidebar:
    st.header("Image Manipulation Options")

    grayscale = st.radio('Conver to grayscale', options=('Human Eye', 'Naive', 'Decomposition','None'),index=3)
    brightness = st.slider("Adjust brightness", min_value=-255, max_value=255, value=0)
    contrast = st.slider("Adjust contrast", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    negative = st.checkbox("Negative")
    binarization_enabled = st.checkbox("Apply binarization")
    if binarization_enabled:
        binarization_threshold = st.slider("Binarization threshold", min_value=0, max_value=255, step=1, value=128)
    average_filter_mask = st.slider("Choose blurr", min_value=1, max_value=100, value=1, step=2)
    edges = st.radio('Choose edge detection method', ('Roberts cross','Sobel filter','None'), index=2)



with st.container():
    col1, col2 = st.columns([1, 1])
    if uploaded_file is None:
        st.warning("Please upload an image to process.")

    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width=400)

    with col2:
        if uploaded_file is not None:
            processed_image = image
            w, h = image.size

            if grayscale == 'Human Eye':
                processed_image = convert_to_gray(processed_image)
            elif grayscale == 'Naive':
                processed_image = convert_to_gray_naive(processed_image)
            elif grayscale =='Decomposition':
                processed_image = convert_to_gray_decomp(processed_image)
            if brightness != 0:
                processed_image = adjust_brightness(processed_image, brightness)
            if contrast != 1.0:
                processed_image = adjust_contrast(processed_image, contrast)
            if negative:
                processed_image = negative_image(processed_image)
            if binarization_enabled:
                processed_image = binarization(processed_image, binarization_threshold)
            if average_filter_mask != 1:
                processed_image = average_filter(processed_image, mask=average_filter_mask)

            if edges == 'Roberts cross':
                processed_image = roberts_cross(processed_image)
            elif edges == "Sobel filter":
                processed_image = sobel_filter(processed_image)

            st.image(processed_image, caption="Processed Image", width=400)

            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG")
            buffered.seek(0)

            st.download_button(
                label="Download Processed Image",
                data=buffered,
                file_name="processed_image.png",
                mime="image/png"
            )

with st.container():
    if uploaded_file is not None:
        st.subheader("Histogram Section")
        show_histogram = st.checkbox("Show Histogram", value=False)

        if show_histogram:
            st.write("Histogram will be shown here:")
            histogram(processed_image, gray_scale=grayscale)

        st.subheader('Horizontal and vertical projection')
        show_proj = st.checkbox('Show plots',value=False)
        if show_proj:
            st.write('Projections plots will be shown')
            projections(processed_image, grayscale = grayscale)

