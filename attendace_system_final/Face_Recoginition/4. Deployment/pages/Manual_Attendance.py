import streamlit as st
import pandas as pd
from datetime import datetime
import glob
import os

st.set_page_config(
    page_title="Attendance System", page_icon="📊", layout="wide"
)

st.title(":writing_hand: Manual Attendance")
st.write("###")

# Ensure the directory exists
base_path = "6. Attendence"
manual_path = os.path.join(base_path, "Manual")
os.makedirs(manual_path, exist_ok=True)

with st.expander("Create New Attendance Sheet"):
    name_of_attendance_sheet = st.text_input("Enter Name of Attendance Sheet")
    if st.button("Create New Attendance Sheet"):
        now = datetime.now()
        date = now.strftime("%d-%m-%Y")
        df = pd.DataFrame(columns=['Date', 'Time', 'Name', 'Status'])
        sheet = f"{name_of_attendance_sheet} {date}"
        file_path = os.path.join(manual_path, f"{sheet}.csv")
        df.to_csv(file_path, index=False)
        st.success(f"Created new attendance sheet: {sheet}")

# datetime object containing current date and time
now = datetime.now()
date = now.strftime("%d-%m-%Y")
time = now.strftime("%H:%M:%S")

tab1, tab2 = st.tabs(["View Sheet", "Add Record"])

with tab1:
    d_l = glob.glob(os.path.join(manual_path, "*.csv"))
    d_l = [os.path.splitext(os.path.basename(file))[0] for file in d_l]
    
    if d_l:
        sheet = st.selectbox("Select Date", d_l)
        if sheet:
            file_path = os.path.join(manual_path, f"{sheet}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.table(df)
            else:
                st.error(f"File not found: {file_path}")
    else:
        st.warning("No attendance sheets found. Please create a new one.")

with tab2:
    if 'sheet' in locals() and sheet:
        file_path = os.path.join(manual_path, f"{sheet}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            name = st.text_input("Enter Student Name")
            status = st.selectbox("Status", ["Present", "Absent"])
            new_row = {'Date': date, 'Time': time, 'Name': name, 'Status': status}
            
            if st.button("Save Record"):
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.to_csv(file_path, index=False)
                st.success("Record Saved")
                st.experimental_rerun()
        else:
            st.error(f"File not found: {file_path}")
    else:
        st.warning("Please select a sheet in the 'View Sheet' tab first.")