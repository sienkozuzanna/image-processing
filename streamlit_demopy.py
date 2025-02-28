import streamlit as st

# widget tekstowy
st.write('Pierwszy tekst na stronie aplikacji')

x = 50
x

# markdown
st.markdown("## Markdown header")

# header subheader
st.header('Streamlit header')

# buttons
is_clicked = st.button('Click me')
if is_clicked:
    st.write('Button is pressed')