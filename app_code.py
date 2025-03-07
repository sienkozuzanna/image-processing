import streamlit as st
from algorithms import *
from algorithms2 import *
from PIL import Image
import numpy as np
import io
import pandas as pd


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


with st.container():
    st.subheader('Image Compression using SVD decomposition')
    if uploaded_file is not None:
        image = processed_image
        w, h = image.size

        k_max_r, k_max_g, k_max_b = get_no_singular_values(image)
        max_k = min(k_max_r, k_max_g, k_max_b)
        #print(max_k)
        num_steps=7
        #orders = np.logspace(np.log10(1), np.log10(max_k), num_steps, base=2, dtype=int)
        #orders = [1, 5, 10, 20, 50, 100, 300, 500, max_k]
        #orders = np.linspace(5, max_k, num_steps, dtype=int)
        orders = np.logspace(np.log10(1), np.log10(max_k), num_steps, dtype=int)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Original Image")
        image_sizes=[]

        for i, order in enumerate(orders):
            compressed_image = compression_svd(image, order)
            axes[i + 1].imshow(compressed_image)
            axes[i + 1].axis('off')
            axes[i + 1].set_title(f"Number of k singular values = {order}")
            compressed_image_pil = Image.fromarray(np.uint8(np.around(compressed_image)))
            compressed_size = get_image_bytes(compressed_image_pil)/1024/1024
            image_sizes.append((order, compressed_size))

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        st.pyplot(fig)

        st.pyplot(visualize_compression_errors(processed_image, orders))

        df_sizes = pd.DataFrame(image_sizes, columns=["Number of k (SVD components)", "Image Size (Mb)"])
        st.table(df_sizes.set_index("Number of k (SVD components)"))



