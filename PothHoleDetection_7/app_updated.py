import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import os
import geocoder
from datetime import datetime
import requests
import math
import folium
import streamlit.components.v1 as components
from streamlit_geolocation import streamlit_geolocation
from PIL import Image

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set page config at the very beginning
st.set_page_config(page_title="Pothole Detection & Alert System", layout="wide")

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c  # in meters

def load_yolo_model():
    weights_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.weights')
    cfg_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.cfg')
    
    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        st.error("Model files not found in utils folder.")
        return None

    net = cv.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
    return model

def process_image(image):
    names_path = os.path.join(BASE_DIR, 'utils', 'obj.names')
    if not os.path.exists(names_path):
        st.error("Class names file not found.")
        return image, pd.DataFrame()

    with open(names_path, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    model = load_yolo_model()
    if model is None:
        return image, pd.DataFrame()

    height, width, _ = image.shape
    image_area = height * width
    total_pothole_area = 0

    classes, scores, boxes = model.detect(image, 0.5, 0.4)

    lat = st.session_state.get('user_lat')
    lon = st.session_state.get('user_lon')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pothole_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        pothole_area = w * h
        total_pothole_area += pothole_area

        severity = "High" if (pothole_area / image_area) > 0.02 else "Medium" if (pothole_area / image_area) > 0.007 else "Low"

        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, f"{label} ({severity})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        pothole_list.append([lat, lon, pothole_area, severity, timestamp])

    pothole_data = pd.DataFrame(pothole_list, columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"])
    return image, pothole_data

def save_pothole_data(pothole_data):
    if pothole_data is None or pothole_data.empty:
        return

    csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
    if os.path.exists(csv_path):
        try:
            existing_data = pd.read_csv(csv_path)
            updated_data = pd.concat([existing_data, pothole_data], ignore_index=True)
            updated_data = updated_data.drop_duplicates()
            updated_data.to_csv(csv_path, index=False)
            st.success(f"Recorded {len(pothole_data)} pothole coordinates in database.")
        except Exception as e:
            st.error(f"Error saving to CSV: {e}")
    else:
        try:
            pothole_data.to_csv(csv_path, index=False)
            st.success(f"Recorded {len(pothole_data)} pothole coordinates in database.")
        except Exception as e:
            st.error(f"Error saving to CSV: {e}")

# ==================== PAGE FUNCTIONS ====================

def image_page():
    st.title("Detect from Image")
    
    if st.session_state.get('user_lat') is None:
        st.warning("Location coordinates are not set. Enable browser location in the sidebar to tag potholes with coordinates.")
        
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="image_uploader_key")
    
    if uploaded_image is not None:
        image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
        
        with st.spinner("Processing..."):
            processed_image, pothole_data = process_image(image)
        
        if not pothole_data.empty:
            st.success(f"Detected {len(pothole_data)} pothole(s)")
            st.table(pothole_data)
            
            height, width, _ = image.shape
            image_area = height * width
            total_pothole_area = pothole_data["Pothole Area (pixels)"].sum()
            st.write(f"Pothole area ratio: {(total_pothole_area/image_area)*100:.2f}% of the frame area.")
            
            st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)
            
            result_path = os.path.join(BASE_DIR, 'pothole_coordinates')
            os.makedirs(result_path, exist_ok=True)
            cv.imwrite(os.path.join(result_path, 'detected_pothole.jpg'), processed_image)
            
            with open(os.path.join(result_path, 'detected_pothole.txt'), 'w') as f:
                for idx, row in pothole_data.iterrows():
                    f.write(f"Location: [{row['Latitude']}, {row['Longitude']}], Area: {row['Pothole Area (pixels)']}, Severity: {row['Severity']}\n")
                f.write(f"\nTotal Pothole Area: {total_pothole_area} pixels\n")
            
            save_pothole_data(pothole_data)
            
            col1, col2 = st.columns(2)
            with col1:
                _, img_encoded = cv.imencode(".jpg", processed_image)
                st.download_button(
                    label="Download Processed Image",
                    data=img_encoded.tobytes(),
                    file_name="detected_pothole.jpg",
                    mime="image/jpeg"
                )
            with col2:
                csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="Download Database (CSV)",
                            data=f,
                            file_name="pothole_data.csv",
                            mime="text/csv"
                        )
        else:
            st.info("No potholes detected.")
            st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)

def video_page():
    st.title("Detect from Video")
    
    if st.session_state.get('user_lat') is None:
        st.warning("Location coordinates are not set. Enable browser location in the sidebar to tag potholes with coordinates.")
        
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_uploader_key")
    
    if uploaded_video is not None:
        temp_video_path = os.path.join(BASE_DIR, "uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        if st.button("Run Detection", key="run_detection_video"):
            with st.spinner("Processing video frames..."):
                names_path = os.path.join(BASE_DIR, 'utils', 'obj.names')
                if not os.path.exists(names_path):
                    st.error("Class names file not found.")
                    return
                
                model = load_yolo_model()
                if model is None:
                    return
                
                cap = cv.VideoCapture(temp_video_path)
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read video.")
                    return
                
                width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                
                result_avi_path = os.path.join(BASE_DIR, 'result.avi')
                result_writer = cv.VideoWriter(result_avi_path, cv.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
                
                lat = st.session_state.get('user_lat')
                lon = st.session_state.get('user_lon')
                pothole_list = []
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                progress_bar = st.progress(0.0)
                frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    classes, scores, boxes = model.detect(frame, 0.5, 0.4)
                    for (classid, score, box) in zip(classes, scores, boxes):
                        label = "pothole"
                        x, y, w, h = box
                        pothole_area = w * h
                        
                        severity = "High" if (pothole_area / (width * height)) > 0.02 else "Medium" if (pothole_area / (width * height)) > 0.007 else "Low"
                        
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv.putText(frame, f"{label} ({severity})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        is_duplicate = False
                        for item in pothole_list:
                            prev_x, prev_y = item[5], item[6]
                            if abs(x - prev_x) < 20 and abs(y - prev_y) < 20:
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
                            pothole_list.append([lat, lon, pothole_area, severity, timestamp, x, y, w, h])
                            
                    result_writer.write(frame)
                    current_frame += 1
                    if frame_count > 0:
                        progress_bar.progress(min(current_frame / frame_count, 1.0))
                        
                cap.release()
                result_writer.release()
                
            st.success("Video processing complete.")
            
            if pothole_list:
                video_potholes = pd.DataFrame(
                    [item[:5] for item in pothole_list],
                    columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"]
                )
                save_pothole_data(video_potholes)
                st.table(video_potholes)
            else:
                st.info("No potholes detected in the video.")
                
            if os.path.exists(result_avi_path):
                with open(result_avi_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video (AVI)",
                        data=file,
                        file_name="processed_video.avi",
                        mime="video/avi"
                    )

def camera_page():
    st.title("Real-time Camera")
    
    if st.session_state.get('user_lat') is None:
        st.warning("Location coordinates are not set. Enable browser location in the sidebar to tag potholes with coordinates.")
        
    captured_file = st.camera_input("Take a photo of the road")
    
    if captured_file is not None:
        image = Image.open(captured_file)
        frame = np.array(image)
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing snapshot..."):
            processed_bgr, pothole_data = process_image(frame_bgr)
            
        if not pothole_data.empty:
            st.success(f"Detected {len(pothole_data)} pothole(s)")
            st.table(pothole_data)
            
            processed_rgb = cv.cvtColor(processed_bgr, cv.COLOR_BGR2RGB)
            st.image(processed_rgb, caption="Processed Image", use_container_width=True)
            
            save_pothole_data(pothole_data)
        else:
            st.info("No potholes detected.")
            processed_rgb = cv.cvtColor(processed_bgr, cv.COLOR_BGR2RGB)
            st.image(processed_rgb, caption="Captured Image", use_container_width=True)

def dashboard_page():
    st.title("Live Map & Alerts")
    
    user_lat = st.session_state.get('user_lat')
    user_lon = st.session_state.get('user_lon')
    loc_source = st.session_state.get('loc_source', 'Unknown')
    
    if user_lat is None or user_lon is None:
        st.warning("Location coordinates are not set. Please use the sidebar to request or enter your coordinates.")
        return

    csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
    if os.path.exists(csv_path):
        try:
            df_potholes = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Error reading database: {e}")
            df_potholes = pd.DataFrame()
    else:
        df_potholes = pd.DataFrame()
        st.info("No pothole database found yet. Run detection on an image first.")

    alert_radius = st.session_state.get('alert_radius', 100)
    enable_audio = st.session_state.get('enable_audio', False)

    st.write(f"Location: {user_lat:.6f}, {user_lon:.6f} (Source: {loc_source})")

    # Center map at user location
    m = folium.Map(location=[user_lat, user_lon], zoom_start=15)

    # User marker
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Current Location",
        tooltip="You are here",
        icon=folium.Icon(color="blue", icon="info-sign", prefix="glyphicon")
    ).add_to(m)

    # Draw alert radius
    folium.Circle(
        location=[user_lat, user_lon],
        radius=alert_radius,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.08,
        tooltip=f"Alert Radius ({alert_radius}m)"
    ).add_to(m)

    nearby_potholes = []
    
    if not df_potholes.empty:
        severity_colors = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red'
        }

        for idx, row in df_potholes.iterrows():
            try:
                p_lat = float(row['Latitude'])
                p_lon = float(row['Longitude'])
                p_sev = row.get('Severity', 'Low')
                p_area = row.get('Pothole Area (pixels)', 0)
                p_time = row.get('Timestamp', 'N/A')
            except Exception:
                continue

            distance = calculate_distance(user_lat, user_lon, p_lat, p_lon)

            if distance <= alert_radius:
                nearby_potholes.append({
                    'index': idx + 1,
                    'distance': distance,
                    'lat': p_lat,
                    'lon': p_lon,
                    'severity': p_sev,
                    'area': p_area,
                    'time': p_time
                })

            popup_text = f"""
            <b>Pothole #{idx+1}</b><br>
            Severity: {p_sev}<br>
            Area: {p_area} px<br>
            Distance: {distance:.1f} m<br>
            Logged: {p_time}
            """
            
            marker_color = severity_colors.get(p_sev, 'blue')
            folium.Marker(
                location=[p_lat, p_lon],
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"Pothole #{idx+1} ({p_sev})",
                icon=folium.Icon(color=marker_color, icon='warning-sign', prefix='glyphicon')
            ).add_to(m)

    # Render folium map
    components.html(m._repr_html_(), height=500)

    # Proximity Alert Banner and Audio warning
    if nearby_potholes:
        nearby_potholes = sorted(nearby_potholes, key=lambda x: x['distance'])
        closest = nearby_potholes[0]
        
        st.markdown(
            f"""
            <div style="background-color: #ff4b4b; padding: 15px; border-radius: 8px; border: 2px solid red; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0;">PROXIMITY WARNING: POTHOLE AHEAD</h3>
                <p style="color: white; font-size: 16px; margin: 8px 0 0 0;">
                    A {closest['severity']} severity pothole is only {closest['distance']:.1f} meters away!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if enable_audio:
            components.html(
                """
                <script>
                (function() {
                    try {
                        const AudioContext = window.AudioContext || window.webkitAudioContext;
                        if (!AudioContext) return;
                        const ctx = new AudioContext();
                        
                        function beep(freq, dur, vol) {
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            osc.type = "sawtooth";
                            osc.frequency.value = freq;
                            gain.gain.setValueAtTime(vol, ctx.currentTime);
                            gain.gain.exponentialRampToValueAtTime(0.00001, ctx.currentTime + dur);
                            osc.start(ctx.currentTime);
                            osc.stop(ctx.currentTime + dur);
                        }
                        
                        beep(880, 0.15, 0.2);
                        setTimeout(() => beep(880, 0.15, 0.2), 220);
                    } catch (e) {
                        console.error("Audio error:", e);
                    }
                })();
                </script>
                """,
                height=0, width=0
            )

        st.write("Potholes inside the Alert Zone:")
        for p in nearby_potholes:
            details_text = f"Pothole #{p['index']} ({p['severity']}) is {p['distance']:.1f}m away (Coordinates: {p['lat']:.5f}, {p['lon']:.5f})"
            if p['severity'] == 'High':
                st.error(details_text)
            elif p['severity'] == 'Medium':
                st.warning(details_text)
            else:
                st.info(details_text)
    else:
        st.success("Clear Road Ahead: No potholes logged within the warning radius.")


# ==================== MAIN INITIALIZER ====================

def main():
    # Set up sidebar location control
    st.sidebar.title("Navigation & Settings")
    
    # Render browser geolocation button in sidebar
    st.sidebar.subheader("Browser Location")
    location = streamlit_geolocation()
    
    # Update coordinates from browser if available
    if location and location.get("latitude") is not None:
        st.session_state['user_lat'] = float(location["latitude"])
        st.session_state['user_lon'] = float(location["longitude"])
        st.session_state['loc_source'] = "Browser GPS"
    
    # Initialize defaults if nothing is set yet
    if 'user_lat' not in st.session_state:
        st.session_state['user_lat'] = None
        st.session_state['user_lon'] = None
        st.session_state['loc_source'] = "Not Connected"
        
    if 'alert_radius' not in st.session_state:
        st.session_state['alert_radius'] = 100
    if 'enable_audio' not in st.session_state:
        st.session_state['enable_audio'] = False

    # Location override methods in the sidebar
    loc_method = st.sidebar.radio(
        "Location Method",
        ["Use Browser GPS", "Use Approximate IP Location", "Manual Coordinates Override"],
        key="fallback_mode_selection"
    )
    
    if loc_method == "Use Browser GPS":
        # If the user selected GPS but it is not available yet, keep it None
        if st.session_state['user_lat'] is None:
            st.sidebar.warning("GPS coordinates are pending. Please click the location button above to allow browser access.")
            
    elif loc_method == "Use Approximate IP Location":
        if 'ip_lat' not in st.session_state:
            # Attempt to fetch approximate IP coordinates
            try:
                res = requests.get("https://ipinfo.io/json", timeout=2)
                if res.status_code == 200:
                    loc_data = res.json()
                    if 'loc' in loc_data:
                        lat, lon = loc_data['loc'].split(',')
                        st.session_state['ip_lat'] = float(lat)
                        st.session_state['ip_lon'] = float(lon)
                        st.session_state['ip_source'] = f"IP Fallback ({loc_data.get('city', 'Approximate')})"
            except Exception:
                try:
                    g = geocoder.ip('me')
                    if g.latlng:
                        st.session_state['ip_lat'] = float(g.latlng[0])
                        st.session_state['ip_lon'] = float(g.latlng[1])
                        st.session_state['ip_source'] = "IP Fallback (Approximate)"
                except Exception:
                    pass
            # If IP lookup failed entirely
            if 'ip_lat' not in st.session_state:
                st.session_state['ip_lat'] = 28.6139
                st.session_state['ip_lon'] = 77.2090
                st.session_state['ip_source'] = "Default (New Delhi)"
                
        st.session_state['user_lat'] = st.session_state['ip_lat']
        st.session_state['user_lon'] = st.session_state['ip_lon']
        st.session_state['loc_source'] = st.session_state['ip_source']
        
    elif loc_method == "Manual Coordinates Override":
        curr_lat = st.session_state.get('user_lat') if st.session_state.get('user_lat') is not None else 28.6139
        curr_lon = st.session_state.get('user_lon') if st.session_state.get('user_lon') is not None else 77.2090
        
        m_lat = st.sidebar.number_input("Latitude", value=curr_lat, format="%.6f", key="man_lat")
        m_lon = st.sidebar.number_input("Longitude", value=curr_lon, format="%.6f", key="man_lon")
        st.session_state['user_lat'] = m_lat
        st.session_state['user_lon'] = m_lon
        st.session_state['loc_source'] = "Manual Entry"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Settings")
    st.session_state['alert_radius'] = st.sidebar.slider(
        "Alert Radius (meters)", 
        min_value=10, 
        max_value=500, 
        value=st.session_state['alert_radius'], 
        step=10
    )
    
    st.session_state['enable_audio'] = st.sidebar.checkbox(
        "Enable Audio Warning Beeps", 
        value=st.session_state['enable_audio']
    )

    # Page setup (No Emojis in titles!)
    pg_image = st.Page(image_page, title="Detect from Image")
    pg_video = st.Page(video_page, title="Detect from Video")
    pg_camera = st.Page(camera_page, title="Real-time Camera")
    pg_map = st.Page(dashboard_page, title="Live Map & Alerts")

    pg = st.navigation([pg_image, pg_video, pg_camera, pg_map])
    pg.run()

if __name__ == "__main__":
    main()
