import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
from deepface import DeepFace

st.set_page_config(page_title="Face Database Manager", page_icon="👤", layout="wide")

st.title("Face Database Management System")
st.write("Add and manage face data for the attendance system")
st.write("---")

# Initialize database directory
DATABASE_PATH = "face_database"
if not os.path.exists(DATABASE_PATH):
    os.makedirs(DATABASE_PATH)

def validate_face_image(image):
    """Validate if the image contains a detectable face."""
    try:
        faces = DeepFace.extract_faces(img_path=image, 
                                     detector_backend='opencv',
                                     enforce_detection=True)
        return len(faces) == 1, faces
    except Exception as e:
        return False, str(e)

def save_uploaded_image(uploaded_file, person_name):
    """Save the uploaded image to the database."""
    person_dir = os.path.join(DATABASE_PATH, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Save the image
    filename = f"{person_name}_{len(os.listdir(person_dir))}.jpg"
    filepath = os.path.join(person_dir, filename)
    cv2.imwrite(filepath, img)
    return filepath

def capture_from_camera(person_name):
    """Capture images from camera."""
    st.write("Camera Preview")
    camera = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    capture_button = st.button("Capture")
    stop_button = st.button("Stop Camera")

    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access camera")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        if capture_button:
            person_dir = os.path.join(DATABASE_PATH, person_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            
            filename = f"{person_name}_{len(os.listdir(person_dir))}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            st.success(f"Image captured and saved as {filename}")
            break

        if stop_button:
            break

    camera.release()
    return capture_button

# Sidebar for database statistics
st.sidebar.title("Database Statistics")
total_people = len(os.listdir(DATABASE_PATH))
st.sidebar.write(f"Total People: {total_people}")

# List all people in database
if total_people > 0:
    st.sidebar.write("People in Database:")
    for person in os.listdir(DATABASE_PATH):
        person_dir = os.path.join(DATABASE_PATH, person)
        num_images = len(os.listdir(person_dir))
        st.sidebar.write(f"- {person}: {num_images} images")

# Main interface
tab1, tab2, tab3 = st.tabs(["Add New Person", "Add to Existing", "Remove Person"])

with tab1:
    st.header("Add New Person")
    new_name = st.text_input("Enter person's name")
    
    if new_name:
        input_method = st.radio("Choose input method", 
                              ["Upload Images", "Use Camera"])
        
        if input_method == "Upload Images":
            uploaded_files = st.file_uploader("Upload face images", 
                                            type=['jpg', 'jpeg', 'png'],
                                            accept_multiple_files=True)
            
            if uploaded_files:
                for file in uploaded_files:
                    # Validate image
                    is_valid, faces = validate_face_image(file)
                    
                    if is_valid:
                        filepath = save_uploaded_image(file, new_name)
                        st.success(f"Successfully saved: {os.path.basename(filepath)}")
                        st.image(file, caption=os.path.basename(filepath), width=200)
                    else:
                        st.error(f"No valid face found in {file.name}. Error: {faces}")
        
        else:  # Use Camera
            st.write("Position your face in front of the camera")
            if capture_from_camera(new_name):
                st.success("Image captured successfully!")

with tab2:
    st.header("Add to Existing Person")
    existing_people = [d for d in os.listdir(DATABASE_PATH) 
                      if os.path.isdir(os.path.join(DATABASE_PATH, d))]
    
    if existing_people:
        selected_person = st.selectbox("Select person", existing_people)
        input_method = st.radio("Choose input method for existing person", 
                              ["Upload Images", "Use Camera"])
        
        if input_method == "Upload Images":
            uploaded_files = st.file_uploader("Upload additional face images", 
                                            type=['jpg', 'jpeg', 'png'],
                                            accept_multiple_files=True,
                                            key="existing_upload")
            
            if uploaded_files:
                for file in uploaded_files:
                    is_valid, faces = validate_face_image(file)
                    
                    if is_valid:
                        filepath = save_uploaded_image(file, selected_person)
                        st.success(f"Successfully saved: {os.path.basename(filepath)}")
                        st.image(file, caption=os.path.basename(filepath), width=200)
                    else:
                        st.error(f"No valid face found in {file.name}. Error: {faces}")
        
        else:  # Use Camera
            st.write("Position your face in front of the camera")
            if capture_from_camera(selected_person):
                st.success("Image captured successfully!")
    else:
        st.warning("No existing people in database. Please add a new person first.")

with tab3:
    st.header("Remove Person")
    if existing_people:
        person_to_remove = st.selectbox("Select person to remove", existing_people)
        if st.button(f"Remove {person_to_remove}"):
            try:
                import shutil
                shutil.rmtree(os.path.join(DATABASE_PATH, person_to_remove))
                st.success(f"Successfully removed {person_to_remove} from database")
                st.rerun()
            except Exception as e:
                st.error(f"Error removing person: {str(e)}")
    else:
        st.warning("No people in database to remove.")

# Display database structure
if st.checkbox("Show Database Structure"):
    st.write("Database Structure:")
    for person in os.listdir(DATABASE_PATH):
        person_dir = os.path.join(DATABASE_PATH, person)
        if os.path.isdir(person_dir):
            st.write(f"└── {person}/")
            for image in os.listdir(person_dir):
                st.write(f"    ├── {image}")