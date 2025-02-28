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
        
        # Kontener z obramowaniem
        with st.container(height=110, border = False):

            uploaded_file = st.file_uploader("Wybierz obraz", type=["jpg", "png", "jpeg"])
            
        with st.container(height=500, border = True):   
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Oryginalny obraz", width=400)



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

                st.image(processed_image, caption="Przerobiony obraz", width=400)
        

    
    
