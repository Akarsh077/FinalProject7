import streamlit as st
import cv2 as cv
import numpy as np
import os
from datetime import datetime
import geocoder
from PIL import Image

# Streamlit UI
st.set_page_config(page_title="Pothole Detection App", layout="centered")
st.title("🛣️ Pothole Detection System")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Upload an image of the road:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    frame = np.array(image)
    
    # Ensure output directory exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(BASE_DIR, "pothole_coordinates")
    os.makedirs(result_path, exist_ok=True)

    # Load YOLO model
    weights_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.weights')
    cfg_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.cfg')
    net = cv.dnn.readNet(weights_path, cfg_path)
    
    # Check if CUDA is available in OpenCV
    use_cuda = False
    try:
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
    except Exception:
        pass

    if use_cuda:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

    height, width, _ = frame.shape
    Conf_threshold = 0.5
    NMS_threshold = 0.4

    # Define ROI (Region of Interest) Mask
    mask = np.zeros_like(frame)
    mask[0:int(0.85 * height), :] = 255
    masked_frame = cv.bitwise_and(frame, mask)

    # Detect potholes
    classes, scores, boxes = model.detect(masked_frame, Conf_threshold, NMS_threshold)
    g = geocoder.ip('me')  # Get location
    pothole_data = []
    total_pothole_area = 0

    for (classid, score, box) in zip(classes, scores, boxes):
        x, y, w, h = box
        recarea = w * h
        total_pothole_area += recarea
        image_area = width * height

        severity = ""
        if score >= 0.7:
            relative_area = recarea / image_area
            if relative_area <= 0.007:
                severity = "Low"
            elif relative_area <= 0.020:
                severity = "Medium"
            else:
                severity = "High"

            # Draw bounding box and label
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{round(score * 100, 2)}% Pothole ({severity})", 
                       (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            pothole_data.append(f"Location: {g.latlng}, Bounding Box: {box}, Severity: {severity}")

    # Save output image
    output_image_path = os.path.join(result_path, 'detected_pothole.jpg')
    cv.imwrite(output_image_path, frame)

    # Write details to a text file
    result_txt_path = os.path.join(result_path, 'detected_pothole.txt')
    with open(result_txt_path, 'w') as f:
        for data in pothole_data:
            f.write(f"{data}\n")
        f.write(f"\nTotal Pothole Area: {total_pothole_area} pixels\n")

    # Show output
    st.image(frame, caption="Detected Potholes", use_column_width=True)
    st.write(f"### 📏 Total Pothole Area: {total_pothole_area} pixels")
    
    # Provide download buttons
    with open(output_image_path, "rb") as file:
        btn = st.download_button(label="📥 Download Processed Image", data=file, file_name="detected_pothole.jpg", mime="image/jpeg")
    
    with open(result_txt_path, "rb") as file:
        btn = st.download_button(label="📄 Download Detection Report", data=file, file_name="detected_pothole.txt", mime="text/plain")

    st.success("Detection Complete! ✅")
