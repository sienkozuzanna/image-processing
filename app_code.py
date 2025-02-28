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

                if grayscale:
                    processed_image = convert_to_gray(image)
                else:
                    processed_image = image
                if(brightness!=0):
                    processed_image = adjust_brightness(processed_image, brightness)
                
                st.image(processed_image, caption="Przerobiony obraz", width=400)
        

    
    
