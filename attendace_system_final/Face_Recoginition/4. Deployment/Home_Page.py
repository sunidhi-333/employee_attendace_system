import streamlit as st
from datetime import datetime
import pandas as pd

# st.set_page_config(
#     page_title="Attendance System", page_icon="📊", layout="wide"
# )
# st.image("logo.png")

home=st.set_page_config(page_title="Home", page_icon=":material/home:"
)
st.title('Attendance system')

st.header("Track and manage attendance with ease !!")

st.write("It will make easy for employee to mark there attendace with 100% accuracy")
st.write(""" 1. You can take attendace with Camera \n 
         \n  2. Add New Student \n
         \n  3. You can also add Manually\n """)


st.button("Contact Us")
