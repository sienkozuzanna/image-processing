import streamlit as st
from algorithms import *
import streamlit as st
from PIL import Image
import io
st.set_page_config(layout="wide")

st.title('Image processing app')
with st.container():
    col1, col2 = st.columns([1,1])

    with col1:
        with st.container(height=110, border = False):
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        with st.container(height=500, border = True):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", width=400)

    with col2:
        with st.container(height=110, border=False):
            pass

        with st.container(height=500, border=True):

            if uploaded_file is not None:
                image = Image.open(uploaded_file)

                grayscale = col1.checkbox("Convert to gray")
                brightness = col1.slider("Adjust brightness", min_value=-255, max_value=255, value=0)
                contrast = col1.slider("Adjust contrast", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
                negative = col1.checkbox("Negative")
                binarization_enabled = col1.checkbox("Apply binarization")

                if grayscale:
                    processed_image = convert_to_gray(image)
                else:
                    processed_image = image
                if(brightness!=0):
                    processed_image = adjust_brightness(processed_image, brightness)
                if(contrast!=0):
                    processed_image = adjust_contrast(processed_image, contrast)
                if negative:
                    processed_image = negative_image(processed_image)
                if binarization_enabled:
                    binarization_threshold = col1.slider("Binarization threshold", min_value=0, max_value=255, step=1, value=128)
                    processed_image=binarization(processed_image, binarization_threshold)



                st.image(processed_image, caption="Processed Image", width=400)

                #downloading processed image
                buffered = io.BytesIO()
                processed_image.save(buffered, format="PNG")
                buffered.seek(0)

                st.download_button(
                    label="Download Processed Image",
                    data=buffered,
                    file_name="processed_image.png",
                    mime="image/png"
                )

            else:
                st.warning("Please upload an image to process.")




