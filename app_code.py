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
    if uploaded_file is not None:
        width, height = Image.open(uploaded_file).size
    else:
        width,height =(0,0)

    grayscale = st.radio('Convert to grayscale', options=('Human Eye', 'Naive', 'Decomposition','None'),index=3)
    brightness = st.slider("Adjust brightness", min_value=-255, max_value=255, value=0)
    contrast = st.slider("Adjust contrast", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    negative = st.checkbox("Negative")
    binarization_enabled = st.checkbox("Apply binarization")
    if binarization_enabled:
        binarization_threshold = st.slider("Binarization threshold", min_value=0, max_value=255, step=1, value=128)
    histogram_equalization_method = st.checkbox('Apply histogram equalization')
    blurr_choice = st.radio('Apply noise reduction', options=('Average filter', 'Median filter', 'Gaussian filter', 'None'), index=3)
    if blurr_choice == 'Average filter':
        avg_mask = st.slider("Choose average filter mask size", min_value=1, max_value=int(min(0.1*width, 0.1*height)), value=1, step=2)
    elif blurr_choice =='Median filter':
        med_mask = st.slider("Choose median filter mask size", min_value=1, max_value=int(min(0.1*width, 0.1*height)), value=1, step=2)
    elif blurr_choice=='Gaussian filter':
        sigma = st.slider('Choose sigma value for Gaussian filter', min_value=1, max_value = 30, value=1, step=1)
    custom_filter_enabled = st.checkbox('Create custom filter')
    if custom_filter_enabled:
        w11 = st.slider('Weight in cell (0,0)', min_value=-9.0, max_value=9.0, step=0.1)
        w12 = st.slider('Weight in cell (0,1)', min_value=-9.0, max_value=9.0, step=0.1)
        w13 = st.slider('Weight in cell (0,2)', min_value=-9.0, max_value=9.0, step=0.1)
        st.write('------------------------')
        w21 = st.slider('Weight in cell (1,0)', min_value=-9.0, max_value=9.0, step=0.1)
        w22 = st.slider('Weight in cell (1,1)', min_value=-9.0, max_value=9.0, step=0.1)
        w23 = st.slider('Weight in cell (1,2)', min_value=-9.0, max_value=9.0, step=0.1)
        st.write('------------------------')
        w31 = st.slider('Weight in cell (2,0)', min_value=-9.0, max_value=9.0, step=0.1)
        w32 = st.slider('Weight in cell (2,1)', min_value=-9.0, max_value=9.0, step=0.1)
        w33 = st.slider('Weight in cell (2,2)', min_value=-9.0, max_value=9.0, step=0.1)

    sharpen_enabled = st.checkbox('Sharpen image')
    if sharpen_enabled:
        sharpen = st.slider('Chose sharpening weight', min_value=9, max_value=30, value=9, step=2)
    edges = st.radio('Choose edge detection method', ('Roberts cross','Sobel filter','None'), index=2)

if uploaded_file is None:
    st.warning("Please upload an image to process.")

with st.container():
    col1, col2 = st.columns([1, 1])
    if uploaded_file is not None:
        with col1:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", width=330)
            
        with col2:
            if uploaded_file is not None:
                processed_image = image
                w, h = image.size

                if grayscale == "None":
                    grayscale = False
                if grayscale == 'Human Eye':
                    processed_image = convert_to_gray(processed_image)
                    grayscale = True
                elif grayscale == 'Naive':
                    processed_image = convert_to_gray_naive(processed_image)
                    grayscale = True
                elif grayscale == 'Decomposition':
                    processed_image = convert_to_gray_decomp(processed_image)
                    grayscale = True
                if brightness != 0:
                    processed_image = adjust_brightness(processed_image, brightness)
                if contrast != 1.0:
                    processed_image = adjust_contrast(processed_image, contrast)
                if negative:
                    processed_image = negative_image(processed_image)
                if binarization_enabled:
                    processed_image = binarization(processed_image, binarization_threshold)
                if blurr_choice == 'Average filter':
                    processed_image = average_filter(processed_image, mask=avg_mask)
                elif blurr_choice == 'Median filter':
                    processed_image = median_filter(processed_image, mask=med_mask)
                elif blurr_choice == 'Gaussian filter':
                    processed_image = gaussian_filter(processed_image, sigma=sigma)
                if custom_filter_enabled:
                    processed_image = custom_filter(processed_image, w11, w12, w13, w21, w22, w23, w31, w32, w33)

                if sharpen_enabled:
                    processed_image = sharpen_filter(processed_image, sharpen)

                if edges == 'Roberts cross':
                    processed_image = roberts_cross(processed_image)
                elif edges == "Sobel filter":
                    processed_image = sobel_filter(processed_image)

                if histogram_equalization_method:
                    processed_image = histogram_equalization(processed_image)

            st.image(processed_image, caption="Processed Image", width=330)

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
        resize_enabled=False
        if uploaded_file is not None:
            st.subheader('Resizing')
            resize_enabled = st.checkbox('Resize photo', value=False)
            if resize_enabled:
                resized_img=processed_image
                width_slider = st.slider('Width:', min_value=0.5, max_value=2.0, step=0.1, value=1.0)
                height_slider = st.slider('Height:', min_value=0.5, max_value=2.0, step=0.1, value=1.0)
                whole_slider = st.slider('Whole picture:', min_value=0.5, max_value=2.0, step=0.1, value=1.0)
                if(width_slider!=1):
                    resized_img = resize_width(processed_image, width_slider)
                if(height_slider!=1):
                    resized_img = resize_height(processed_image, height_slider)
                if(whole_slider!=1):
                    resized_img = resize_whole(processed_image, whole_slider)
with st.container():
    col1, col2,col3 = st.columns([1, 5,1])
    with col2:
        if not resize_enabled:
            resized_img=None
        if resized_img is not None:
            st.image(resized_img, caption="Resized Image")
with st.container():
    st.subheader("Histogram Section")
    show_histogram = st.checkbox("Show Histogram", value=False)

    if show_histogram:
        st.write("Histogram will be shown here:")
        histogram(processed_image, gray_scale=grayscale)

    st.subheader('Horizontal and vertical projection')
    show_proj = st.checkbox('Show plots',value=False)
    if show_proj:
        st.write('Projections plots will be shown')
        projections(processed_image)


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



