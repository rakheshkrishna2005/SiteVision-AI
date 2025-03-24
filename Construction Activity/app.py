import streamlit as st
import cv2
import numpy as np
import tempfile as temp_file_module
from ultralytics import YOLO
import torch
import time
import os
import csv
from dotenv import load_dotenv
from datetime import datetime

# Load the .env file
load_dotenv()

# Get the URL from the .env file
CAMERA_URL = os.getenv("CAMERA_URL")

CLASSES = ['Paver', 'Compactor', 'Bulldozer', 'Excavator', 'Concrete Mixture', 'Crane', 'Dump Truck', 'Utility Ducts', 'Concrete Pumps']

EQUIPMENT_INFO = {
    'Excavator': {
        'activity': 'Excavation and Earth Moving',
        'stage': 'Foundation and Site Preparation'
    },
    'Bulldozer': {
        'activity': 'Grading and Levelling (including road construction)',
        'stage': 'Foundation and Site Preparation, Road Construction'
    },
    'Compactor': {
        'activity': 'Soil and Surface Compaction (including road construction)',
        'stage': 'Foundation and Site Preparation, Road Construction'
    },
    'Concrete Mixture': {
        'activity': 'Concrete Mixing and Pouring',
        'stage': 'Structural Construction'
    },
    'Concrete Pumps': {
        'activity': 'Concrete Placement',
        'stage': 'Structural Construction'
    },
    'Crane': {
        'activity': 'Material Lifting and Placement',
        'stage': 'Structural Construction'
    },
    'Dump Truck': {
        'activity': 'Material Transportation',
        'stage': 'Foundation and Site Preparation'
    },
    'Utility Ducts': {
        'activity': 'Utility Installation',
        'stage': 'MEP Installation'
    },
    'Paver': {
        'activity': 'Road Surface Construction',
        'stage': 'Infrastructure Development'
    }
}

def detect_objects(image, model, classes, conf):
    results = model(image, conf=conf, classes=classes)
    annotated_image = results[0].plot()
    return annotated_image, results[0].boxes.cls.tolist()

def is_cuda_available():
    return torch.cuda.is_available()

def save_and_download_image(image):
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".jpg", image_bgr)
        if not is_success:
            st.error("Failed to encode image")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_image_{timestamp}.jpg"
        
        st.markdown(
            """
            <style>
            .st-emotion-cache-1vt4y43 {
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
                width: fit-content !important;
            }
            .st-emotion-cache-12118b6 {
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
                width: fit-content !important;
            }
            .st-emotion-cache-15hul6a {
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
                width: fit-content !important;
            }
            .stSuccess {
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.download_button(
            label="Download Processed Image",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/jpeg"
        )
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def save_and_download_video(frames):
    try:
        if not frames:
            st.error("No frames to save")
            return
        with temp_file_module.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_file.name, fourcc, 30.0, (width, height))
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            with open(temp_video_file.name, 'rb') as f:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_video_{timestamp}.mp4"
                
                st.markdown(
                    """
                    <style>
                    .st-emotion-cache-1vt4y43 {
                        display: block !important;
                        margin-left: auto !important;
                        margin-right: auto !important;
                        width: fit-content !important;
                    }
                    .st-emotion-cache-12118b6 {
                        display: block !important;
                        margin-left: auto !important;
                        margin-right: auto !important;
                        width: fit-content !important;
                    }
                    .st-emotion-cache-15hul6a {
                        display: block !important;
                        margin-left: auto !important;
                        margin-right: auto !important;
                        width: fit-content !important;
                    }
                    .stSuccess {
                        text-align: center;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                st.download_button(
                    label="Download Processed Video",
                    data=f.read(),
                    file_name=filename,
                    mime="video/mp4"
                )
                
        if os.path.exists(temp_video_file.name):
            try:
                os.unlink(temp_video_file.name)
            except PermissionError:
                pass
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        
def save_detection_data_to_csv(object_data, project_name, site_location, date, unique_key):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_report_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            # Prepare CSV headers
            fieldnames = ['Project Name', 'Site Location', 'Date', 
                          'Construction Equipment', 'Activity In Progress', 
                          'Construction Stage', 'Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write headers
            writer.writeheader()
            
            # Write project details and detection data
            for item in object_data:
                writer.writerow({
                    'Project Name': project_name,
                    'Site Location': site_location,
                    'Date': date,
                    **item
                })
        
        with open(filename, 'rb') as f:
            st.download_button(
                label="Download CSV Report",
                data=f,
                file_name=filename,
                mime="text/csv",
                key=f"csv_download_{unique_key}"
            )
        
        # Clean up the temporary CSV file
        os.unlink(filename)
    except Exception as e:
        st.error(f"Error creating CSV: {str(e)}")

def main():
    st.set_page_config(page_title="SiteVision Activity AI", page_icon="🏗️")
    
    st.markdown("<h1 style='text-align: center;'>🏗️ SiteVision AI", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; padding-bottom: 20px;">
        <a href=""><img src="https://img.shields.io/badge/Progress Tracking-Open-blue" style="margin-right: 10px;"></a>
        <a href=""><img src="https://img.shields.io/badge/Progress Map-Open-blue"></a>
    </div>
    """, unsafe_allow_html=True)
    
    with open('main.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.sidebar.title("⚙️ Settings")
    st.markdown("""
                <style>
                .stButton > button {
                    width: 100%;
                }
                </style>""", 
                unsafe_allow_html=True)
    st.sidebar.markdown('<div class="settings-container">', unsafe_allow_html=True)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.sidebar.markdown("---")
    
    st.sidebar.title("📋 Project Details")
    project_name = st.sidebar.text_input("Project Name")
    site_location = st.sidebar.text_input("Site Location")
    date = st.sidebar.date_input("Date", datetime.now())
    st.sidebar.markdown("---")
    
    cuda_available = is_cuda_available()

    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_video = st.sidebar.file_uploader("Upload Files", type=['mp4', 'jpg', 'jpeg'])
    
    use_webcam = st.sidebar.button("Use Camera")
    
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")
        st.sidebar.info("To enable GPU acceleration, make sure you have a CUDA-capable GPU and PyTorch is installed with CUDA support.")
    
    model = YOLO('best.pt')
        
    selected_classes = st.sidebar.multiselect("Select Classes", CLASSES, default=['Crane'])
    class_indices = [CLASSES.index(cls) for cls in selected_classes]
    
    enable_gpu = st.sidebar.checkbox("🤖 Enable GPU", value=False, disabled=not cuda_available)
    if enable_gpu and cuda_available:
        model.to('cuda')
        st.sidebar.success("GPU enabled successfully!")
    else:
        model.to('cpu')
        st.sidebar.info("Using CPU for processing.")
    
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    object_count_placeholder = st.empty()

    st.markdown("""
    <style>
    .detected-object-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
    }
    .detected-object-table th, .detected-object-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .detected-object-table th {
        background-color: var(--background-color);
    }
    .detected-object-table tr:nth-child(even) {
        background-color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)

    processed_frames = []

    def save_detection_data_to_csv(object_data, project_name, site_location, date):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                # Prepare CSV headers
                fieldnames = ['Project Name', 'Site Location', 'Date', 
                              'Construction Equipment', 'Activity In Progress', 
                              'Construction Stage', 'Count']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write headers
                writer.writeheader()
                
                # Write project details and detection data
                for item in object_data:
                    writer.writerow({
                        'Project Name': project_name,
                        'Site Location': site_location,
                        'Date': date,
                        **item
                    })
            
            with open(filename, 'rb') as f:
                st.download_button(
                    label="Download CSV Report",
                    data=f,
                    file_name=filename,
                    mime="text/csv"
                )
            
            # Clean up the temporary CSV file
            os.unlink(filename)
        except Exception as e:
            st.error(f"Error creating CSV: {str(e)}")

    if use_webcam:
        cap = cv2.VideoCapture(CAMERA_URL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frame from the phone camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [
                {
                    "Construction Equipment": CLASSES[int(cls)],
                    "Activity In Progress": EQUIPMENT_INFO[CLASSES[int(cls)]]['activity'],
                    "Construction Stage": EQUIPMENT_INFO[CLASSES[int(cls)]]['stage'],
                    "Count": count
                } 
                for cls, count in zip(unique_classes, counts)
            ]
            
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Construction Equipment</th><th>Activity In Progress</th><th>Construction Stage</th><th>Count</th></tr>" +
                "".join([
                    f"<tr><td>{item['Construction Equipment']}</td>" +
                    f"<td>{item['Activity In Progress']}</td>" +
                    f"<td>{item['Construction Stage']}</td>" +
                    f"<td>{item['Count']}</td></tr>" 
                    for item in object_data
                ]) +
                "</table>",
                unsafe_allow_html=True
            )

            if object_data:
                save_detection_data_to_csv(
                    object_data, 
                    project_name, 
                    site_location, 
                    date.strftime("%Y-%m-%d")
                )

            if not use_webcam:
                break

        cap.release()
        
        if processed_frames:
            save_and_download_video(processed_frames)

    elif uploaded_video is not None:
        is_video = uploaded_video.type.startswith('video/')
        
        if is_video:
            temp_video_file = temp_file_module.NamedTemporaryFile(delete=False)
            try:
                temp_video_file.write(uploaded_video.read())
                temp_video_file.close()

                vf = cv2.VideoCapture(temp_video_file.name)
                
                try:
                    while vf.isOpened():
                        ret, frame = vf.read()
                        if not ret:
                            break

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
                        video_placeholder.image(annotated_frame, channels="RGB")
                        processed_frames.append(annotated_frame)

                        unique_classes, counts = np.unique(detected_classes, return_counts=True)
                        object_data = [
                            {
                                "Construction Equipment": CLASSES[int(cls)],
                                "Activity In Progress": EQUIPMENT_INFO[CLASSES[int(cls)]]['activity'],
                                "Construction Stage": EQUIPMENT_INFO[CLASSES[int(cls)]]['stage'],
                                "Count": count
                            } 
                            for cls, count in zip(unique_classes, counts)
                        ]
                        
                        object_count_placeholder.markdown(
                            "<table class='detected-object-table'>" +
                            "<tr><th>Construction Equipment</th><th>Activity In Progress</th><th>Construction Stage</th><th>Count</th></tr>" +
                            "".join([
                                f"<tr><td>{item['Construction Equipment']}</td>" +
                                f"<td>{item['Activity In Progress']}</td>" +
                                f"<td>{item['Construction Stage']}</td>" +
                                f"<td>{item['Count']}</td></tr>" 
                                for item in object_data
                            ]) +
                            "</table>",
                            unsafe_allow_html=True
                        )

                    if object_data:
                        save_detection_data_to_csv(
                            object_data, 
                            project_name, 
                            site_location, 
                            date.strftime("%Y-%m-%d")
                        )

                finally:
                    vf.release()
                    
                if processed_frames:
                    save_and_download_video(processed_frames)

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            
            finally:
                try:
                    if 'vf' in locals():
                        vf.release()
                    if os.path.exists(temp_video_file.name):
                        try:
                            os.unlink(temp_video_file.name)
                        except PermissionError:
                            pass
                except Exception as e:
                    st.warning(f"Warning: Could not clean up temporary files: {str(e)}")
        else:
            image = cv2.imdecode(np.frombuffer(uploaded_video.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotated_image, detected_classes = detect_objects(image, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_image, channels="RGB")
            
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [
                {
                    "Construction Equipment": CLASSES[int(cls)],
                    "Activity In Progress": EQUIPMENT_INFO[CLASSES[int(cls)]]['activity'],
                    "Construction Stage": EQUIPMENT_INFO[CLASSES[int(cls)]]['stage'],
                    "Count": count
                } 
                for cls, count in zip(unique_classes, counts)
            ]
            
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Construction Equipment</th><th>Activity In Progress</th><th>Construction Stage</th><th>Count</th></tr>" +
                "".join([
                    f"<tr><td>{item['Construction Equipment']}</td>" +
                    f"<td>{item['Activity In Progress']}</td>" +
                    f"<td>{item['Construction Stage']}</td>" +
                    f"<td>{item['Count']}</td></tr>" 
                    for item in object_data
                ]) +
                "</table>",
                unsafe_allow_html=True
            )
            
            if object_data:
                save_detection_data_to_csv(
                    object_data, 
                    project_name, 
                    site_location, 
                    date.strftime("%Y-%m-%d")
                )
            
            save_and_download_image(annotated_image)

if __name__ == "__main__":
    main()