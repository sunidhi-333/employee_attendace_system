import csv
import streamlit as st
from PIL import Image
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Attendance System", page_icon="📊", layout="wide")

st.title(":camera: Attendance with Camera")
st.write("---")


base_path = "face_database"
attendance_path = os.path.join("6. Attendence", "With_Camera")
os.makedirs(base_path, exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)

# Load face database
@st.cache_resource
def load_face_database(database_path):
    face_database = {}
    if not os.path.exists(database_path):
        # st.error(f"Database directory '{database_path}' does not exist!")
        return face_database
    items = os.listdir(database_path)
    if not items:
        st.error(f"Database directory '{database_path}' is empty! Please add some face images.")
        return face_database

    for person_name in items:
        person_path = os.path.join(database_path, person_name)
        if os.path.isdir(person_path):
            # st.info(f"Processing directory for {person_name}")
            
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                st.warning(f"No images found for {person_name}")
                continue
            
            for image_name in images:
                image_path = os.path.join(person_path, image_name)
                try:
                    embedding = DeepFace.represent(img_path=image_path, 
                                                 model_name="OpenFace",  
                                                 enforce_detection=False)
                    
                    if embedding:
                        if person_name not in face_database:
                            face_database[person_name] = []
                        face_database[person_name].append(embedding[0]["embedding"])
                    else:
                        st.warning(f"No embedding generated for {image_path}")
                        
                except Exception as e:
                    st.warning(f"Error processing {image_path}: {str(e)}")
                    continue
    
    if not face_database:
        st.error("No face encodings were successfully generated!")
        
    return face_database

try:
    face_database = load_face_database(base_path)
    if not face_database:
        st.error("No face encodings found in the database. Please add face images to the database.")
        st.stop()
except Exception as e:
    st.error(f"Error loading face database: {str(e)}")
    st.stop()

def find_closest_match(embedding, face_database, threshold=0.35):
    max_similarity = -1
    closest_match = "Unknown"
    
    for person_name, embeddings in face_database.items():
        for stored_embedding in embeddings:
            similarity = cosine_similarity(
                np.array(embedding).reshape(1, -1),
                np.array(stored_embedding).reshape(1, -1)
            )[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > threshold:
                    closest_match = person_name
    
    return closest_match, max_similarity

def detect_known_faces(frame, face_database):
    recognized_faces = []
    try:
        faces = DeepFace.extract_faces(img_path=frame, 
                                     detector_backend='opencv',
                                     enforce_detection=False)
        
        for face in faces:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            face_image = frame[y:y+h, x:x+w]
            embedding = DeepFace.represent(img_path=face_image, 
                                         model_name="OpenFace", 
                                         enforce_detection=False)
            
            if embedding:
                person_name, similarity = find_closest_match(embedding[0]["embedding"], 
                                                           face_database)
                recognized_faces.append({
                    'location': (x, y, w, h),
                    'name': person_name,
                    'similarity': similarity
                })
    except Exception as e:
        st.warning(f"Error in face detection: {str(e)}")
    
    return recognized_faces

def get_enrolled_students():
    return list(face_database.keys())


def create_daily_attendance_sheet():
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    da = now.strftime("%d-%B-%Y")
    
    enrolled_students = sorted(get_enrolled_students())
    

    df = pd.DataFrame({
        'Date': [date] * len(enrolled_students),
        'Name': enrolled_students,
        'Status': ['Absent'] * len(enrolled_students),
        'Time': ['<N/A>'] * len(enrolled_students),
        'Confidence': [0.0] * len(enrolled_students)
    })

    attendance_sheet_name = f"Attendance {da}"
    file_path = os.path.join(attendance_path, f"{attendance_sheet_name}.csv")

    if not os.path.exists(file_path):
        df = df[['Date', 'Name', 'Status', 'Time', 'Confidence']]
        df.to_csv(file_path, index=False)
        st.success(f"Created new attendance sheet: {attendance_sheet_name}")
    
    return attendance_sheet_name


def update_attendance(attendance_path, name_of_attendance_sheet, name, time, confidence):
    file_path = os.path.join(attendance_path, f"{name_of_attendance_sheet}.csv")
    temp_file_path = os.path.join(attendance_path, f"{name_of_attendance_sheet}_temp.csv")

    fieldnames = ['Date', 'Name', 'Status', 'Time', 'Confidence']

    try:
        
        with open(file_path, 'r') as csv_file, open(temp_file_path, 'w', newline='') as temp_file:
            reader = csv.DictReader(csv_file)
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if row['Name'].lower() == name.lower():
                    current_confidence = float(row.get('Confidence', 0))
                    if confidence > current_confidence:
                        row.update({
                            'Status': 'Present',
                            'Time': time,
                            'Confidence': f"{confidence:.2f}"
                        })
                writer.writerow(row)
        os.replace(temp_file_path, file_path)
        
    except Exception as e:
        st.error(f"Error updating attendance: {str(e)}")
      
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

name_of_attendance_sheet = create_daily_attendance_sheet()


with st.expander("View Sheets"):
    d_l = glob.glob(os.path.join(attendance_path, "*.csv"))
    d_l = [os.path.splitext(os.path.basename(file))[0] for file in d_l]
    if d_l:
        selected_sheet = st.selectbox("Select Sheet", d_l)
    else:
        st.warning("No attendance sheets found.")


FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
start = st.button("Start Taking Attendance")
stop = st.button("Stop Taking Attendance")

while start:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to grab frame")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    recognized_faces = detect_known_faces(frame_rgb, face_database)
    
    for face in recognized_faces:
        x, y, w, h = face['location']
        name = face['name']
        similarity = face['similarity']
        
        color = (0, 255, 0) if similarity > 0.65 else (0, 0, 255)
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_rgb, f"{name} ({similarity:.2f})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if name != "Unknown":
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            update_attendance(attendance_path, name_of_attendance_sheet, name, time, similarity)
    
    FRAME_WINDOW.image(frame_rgb)
    
    if stop:
        break

if stop:
    camera.release()
    cv2.destroyAllWindows()


df = pd.read_csv(os.path.join(attendance_path, f"{name_of_attendance_sheet}.csv"))
st.table(df)