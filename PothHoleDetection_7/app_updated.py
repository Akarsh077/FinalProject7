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

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
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
    """
    Loads the YOLOv4 Tiny model and configures it using absolute paths.
    """
    weights_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.weights')
    cfg_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.cfg')
    
    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        st.error(f"Model files not found! Ensure yolov4_tiny.weights and yolov4_tiny.cfg are in {os.path.join(BASE_DIR, 'utils')}")
        return None

    net = cv.dnn.readNet(weights_path, cfg_path)

    # Streamlit Cloud runs on CPU-only. We force OpenCV CPU backend.
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
    return model

def process_image(image):
    """
    Detects potholes on the input image using YOLOv4.
    """
    names_path = os.path.join(BASE_DIR, 'utils', 'obj.names')
    if not os.path.exists(names_path):
        st.error("Class names file 'obj.names' not found in utils folder.")
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

    # Get location and timestamp from session state or fallback
    lat = st.session_state.get('user_lat', 28.6139)
    lon = st.session_state.get('user_lon', 77.2090)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pothole_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        pothole_area = w * h
        total_pothole_area += pothole_area

        severity = "High" if (pothole_area / image_area) > 0.02 else "Medium" if (pothole_area / image_area) > 0.007 else "Low"

        # Draw box and labels
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, f"{label} ({severity})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        pothole_list.append([lat, lon, pothole_area, severity, timestamp])

    pothole_data = pd.DataFrame(pothole_list, columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"])
    return image, pothole_data

def save_pothole_data(pothole_data):
    """
    Appends pothole detections to pothole_data.csv.
    """
    if pothole_data is None or pothole_data.empty:
        return

    csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
    if os.path.exists(csv_path):
        try:
            existing_data = pd.read_csv(csv_path)
            updated_data = pd.concat([existing_data, pothole_data], ignore_index=True)
            updated_data = updated_data.drop_duplicates()
            updated_data.to_csv(csv_path, index=False)
            st.success(f"Successfully recorded {len(pothole_data)} new pothole location(s) in the database.")
        except Exception as e:
            st.error(f"Error appending data to CSV database: {e}")
    else:
        try:
            pothole_data.to_csv(csv_path, index=False)
            st.success(f"Created new database and recorded {len(pothole_data)} pothole location(s).")
        except Exception as e:
            st.error(f"Error creating CSV database: {e}")

# ==================== PAGE FUNCTIONS ====================

def home_page():
    st.title("🏠 Pothole Detection & Alert System")
    st.write("---")
    
    st.markdown("""
    Welcome to the **Pothole Detection & Alert System**! This application uses a custom trained 
    YOLOv4-Tiny model to detect road potholes and map their coordinates for proximity warning alerts.
    
    ### ⚙️ Features Available:
    1. **Detect from Image**: Upload static photos of roads to detect potholes and log them.
    2. **Detect from Video**: Process road trip videos frame-by-frame.
    3. **Real-time Camera**: Use your device's browser camera to snap road conditions directly.
    4. **Live Map & Alerts Dashboard**: View detected potholes on an interactive map and receive real-time proximity alarms (including audio warnings) as you approach them.
    
    ### 📍 Current Navigation Location Status:
    """)
    
    # Display coordinates status in dashboard
    user_lat = st.session_state.get('user_lat', 28.6139)
    user_lon = st.session_state.get('user_lon', 77.2090)
    loc_source = st.session_state.get('loc_source', 'Default')
    
    st.info(f"🧭 **Active Location**: Latitude `{user_lat:.6f}`, Longitude `{user_lon:.6f}`  \n*(Source: {loc_source})*")
    
    st.markdown("""
    > [!TIP]
    > To obtain the most accurate coordinates, click the **Get Browser Location** button in the sidebar and allow location permissions. If your browser blocks permissions, you can switch to **Manual Coordinates Override** in the sidebar to simulate location movement!
    """)

def image_page():
    st.title("📷 Detect Potholes from Image")
    st.write("Upload a road photo to analyze it for potholes using our YOLO model.")
    
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="image_uploader_key")
    
    if uploaded_image is not None:
        # Read uploaded image
        image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
        
        st.subheader("Processing Result")
        with st.spinner("Running YOLO detection..."):
            processed_image, pothole_data = process_image(image)
        
        if not pothole_data.empty:
            st.success(f"Detected {len(pothole_data)} pothole(s)!")
            st.table(pothole_data)
            
            # Show details of detections
            height, width, _ = image.shape
            image_area = height * width
            total_pothole_area = pothole_data["Pothole Area (pixels)"].sum()
            st.write(f"**Pothole Area Ratio**: {(total_pothole_area/image_area)*100:.2f}% of the frame area.")
            
            # Display processed image
            st.image(processed_image, channels="BGR", caption="Processed Image with Detections", use_container_width=True)
            
            # Save results into files as in the original app
            result_path = os.path.join(BASE_DIR, 'pothole_coordinates')
            os.makedirs(result_path, exist_ok=True)
            cv.imwrite(os.path.join(result_path, 'detected_pothole.jpg'), processed_image)
            
            with open(os.path.join(result_path, 'detected_pothole.txt'), 'w') as f:
                for idx, row in pothole_data.iterrows():
                    f.write(f"Location: [{row['Latitude']}, {row['Longitude']}], Area: {row['Pothole Area (pixels)']}, Severity: {row['Severity']}\n")
                f.write(f"\nTotal Pothole Area: {total_pothole_area} pixels\n")
            
            # Append detections to the shared database
            save_pothole_data(pothole_data)
            
            # Provide downloads
            col1, col2 = st.columns(2)
            with col1:
                _, img_encoded = cv.imencode(".jpg", processed_image)
                st.download_button(
                    label="📥 Download Processed Image",
                    data=img_encoded.tobytes(),
                    file_name="detected_pothole.jpg",
                    mime="image/jpeg"
                )
            with col2:
                # Re-download full database CSV
                csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Updated Database (CSV)",
                            data=f,
                            file_name="pothole_data.csv",
                            mime="text/csv"
                        )
        else:
            st.info("No potholes detected in this image.")
            st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)

def video_page():
    st.title("🎥 Detect Potholes from Video")
    st.write("Upload a dashboard video file to run frame-by-frame detection.")
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_uploader_key")
    
    if uploaded_video is not None:
        temp_video_path = os.path.join(BASE_DIR, "uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        if st.button("Run Detection Process", key="run_detection_video"):
            with st.spinner("Processing video frames... This may take a while depending on length."):
                
                # Setup YOLO
                names_path = os.path.join(BASE_DIR, 'utils', 'obj.names')
                if not os.path.exists(names_path):
                    st.error("Class names file 'obj.names' not found in utils folder.")
                    return
                
                model = load_yolo_model()
                if model is None:
                    return
                
                cap = cv.VideoCapture(temp_video_path)
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read video. Ensure it is a valid format.")
                    return
                
                width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                
                # Write output avi file
                result_avi_path = os.path.join(BASE_DIR, 'result.avi')
                result_writer = cv.VideoWriter(result_avi_path, cv.VideoWriter_fourcc(*'MJPG'), 10, (width, height))
                
                lat = st.session_state.get('user_lat', 28.6139)
                lon = st.session_state.get('user_lon', 77.2090)
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
                        
                        # Prevent logging duplicate potholes close to each other in the same video run
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
                
            st.success("Video processing complete!")
            
            # Save data to CSV
            if pothole_list:
                video_potholes = pd.DataFrame(
                    [item[:5] for item in pothole_list],
                    columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"]
                )
                save_pothole_data(video_potholes)
                st.table(video_potholes)
            else:
                st.info("No potholes were detected in the video.")
                
            # Provide download option
            if os.path.exists(result_avi_path):
                with open(result_avi_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Video (AVI)",
                        data=file,
                        file_name="processed_video.avi",
                        mime="video/avi"
                    )

def camera_page():
    st.title("📹 Real-time Camera Detection")
    st.write("Use your device's camera to take a photo of the road for instant pothole detection.")
    
    # st.camera_input is standard, works natively in the browser on mobile and desktop
    captured_file = st.camera_input("Take a photo of the road")
    
    if captured_file is not None:
        # Read the captured image
        image = Image.open(captured_file)
        frame = np.array(image)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        st.subheader("Processing Result")
        with st.spinner("Analyzing snapshot..."):
            processed_bgr, pothole_data = process_image(frame_bgr)
            
        if not pothole_data.empty:
            st.success(f"Detected {len(pothole_data)} pothole(s)!")
            st.table(pothole_data)
            
            # Convert BGR back to RGB for Streamlit rendering
            processed_rgb = cv.cvtColor(processed_bgr, cv.COLOR_BGR2RGB)
            st.image(processed_rgb, caption="Processed Image", use_container_width=True)
            
            # Save detection
            save_pothole_data(pothole_data)
        else:
            st.info("No potholes detected in this camera frame.")
            processed_rgb = cv.cvtColor(processed_bgr, cv.COLOR_BGR2RGB)
            st.image(processed_rgb, caption="Captured Image", use_container_width=True)

def dashboard_page():
    st.title("🗺️ Live Map & Proximity Alerts")
    st.write("Interactive map displaying all detected potholes. Alerts trigger if you are near any logged pothole.")
    
    # Load pothole database
    csv_path = os.path.join(BASE_DIR, "pothole_data.csv")
    if os.path.exists(csv_path):
        try:
            df_potholes = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Error reading database: {e}")
            df_potholes = pd.DataFrame()
    else:
        df_potholes = pd.DataFrame()
        st.info("No pothole database found yet. Run detection on an image, video, or camera snapshot first to log coordinates.")

    # Get location settings from session state
    user_lat = st.session_state.get('user_lat', 28.6139)
    user_lon = st.session_state.get('user_lon', 77.2090)
    loc_source = st.session_state.get('loc_source', 'Default')
    alert_radius = st.session_state.get('alert_radius', 100)
    enable_audio = st.session_state.get('enable_audio', False)

    st.write(f"📍 **Your Location**: `{user_lat:.6f}, {user_lon:.6f}` *(Source: {loc_source})*")

    # Center map at user location
    m = folium.Map(location=[user_lat, user_lon], zoom_start=15)

    # Add blue marker for user
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Current Location",
        tooltip="You are here",
        icon=folium.Icon(color="blue", icon="info-sign", prefix="glyphicon")
    ).add_to(m)

    # Draw alert radius circle
    folium.Circle(
        location=[user_lat, user_lon],
        radius=alert_radius,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.08,
        tooltip=f"Alert Radius ({alert_radius}m)"
    ).add_to(m)

    # Check for proximity and display marker coordinates
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

            # Store potholes inside warning radius
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
            Area: {p_area} pixels<br>
            Distance: {distance:.1f} meters<br>
            Logged: {p_time}
            """
            
            marker_color = severity_colors.get(p_sev, 'blue')
            folium.Marker(
                location=[p_lat, p_lon],
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"Pothole #{idx+1} ({p_sev})",
                icon=folium.Icon(color=marker_color, icon='warning-sign', prefix='glyphicon')
            ).add_to(m)

    # Render folium map in the page
    components.html(m._repr_html_(), height=500)

    # Trigger alerts
    if nearby_potholes:
        # Sort by distance (closest first)
        nearby_potholes = sorted(nearby_potholes, key=lambda x: x['distance'])
        closest = nearby_potholes[0]
        
        # Display flashing red alarm banner
        st.markdown(
            f"""
            <div style="background-color: #ff4b4b; padding: 15px; border-radius: 8px; border: 2px solid red; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0;">🚨 PROXIMITY ALERT: POTHOLE AHEAD!</h3>
                <p style="color: white; font-size: 16px; margin: 8px 0 0 0;">
                    A <b>{closest['severity']}</b> severity pothole is only <b>{closest['distance']:.1f} meters</b> away!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Trigger Synthesized Audio Alarm if toggled ON in sidebar
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
                        
                        // Play a alarm double-beep
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
            st.info("🔊 Warning audio alert active. Click on page if you don't hear the alarm sound.")

        # Show list of nearby items
        st.write("### ⚠️ Potholes inside the Alert Zone:")
        for p in nearby_potholes:
            details_text = f"**Pothole #{p['index']}** ({p['severity']}) is **{p['distance']:.1f}m** away (Coordinates: {p['lat']:.5f}, {p['lon']:.5f})"
            if p['severity'] == 'High':
                st.error(details_text)
            elif p['severity'] == 'Medium':
                st.warning(details_text)
            else:
                st.info(details_text)
    else:
        st.success("✅ **Clear Road Ahead**: No potholes logged within the warning radius.")


# ==================== MAIN INITIALIZER ====================

def main():
    # Initialize session state coordinates & preferences
    if 'user_lat' not in st.session_state:
        st.session_state['user_lat'] = 28.6139
    if 'user_lon' not in st.session_state:
        st.session_state['user_lon'] = 77.2090
    if 'loc_source' not in st.session_state:
        st.session_state['loc_source'] = "Default (New Delhi)"
    if 'alert_radius' not in st.session_state:
        st.session_state['alert_radius'] = 100
    if 'enable_audio' not in st.session_state:
        st.session_state['enable_audio'] = False  # DEFAULT OFF AS REQUESTED!

    # SIDEBAR: Geolocation Controller & Settings
    st.sidebar.title("🧭 Navigation Control")
    
    # 1. Streamlit Geolocation Browser Component
    # This renders a small button. When clicked, browser prompts for GPS coordinates
    st.sidebar.subheader("Browser Location")
    location = streamlit_geolocation()
    
    if location and location.get("latitude") is not None:
        st.session_state['user_lat'] = float(location["latitude"])
        st.session_state['user_lon'] = float(location["longitude"])
        st.session_state['loc_source'] = "Browser GPS (Precise)"
    else:
        loc_method = st.sidebar.radio(
            "Fallback/Override Mode",
            ["IP Address Fallback", "Manual Coordinates Override"],
            key="fallback_mode_selection"
        )
        
        if loc_method == "IP Address Fallback":
            # Cache the IP location so we don't spam requests on every Streamlit rerun
            if 'ip_lat' not in st.session_state:
                st.session_state['ip_lat'] = 28.6139
                st.session_state['ip_lon'] = 77.2090
                st.session_state['ip_source'] = "Default (New Delhi)"
                
                try:
                    # Request client geo from ipinfo
                    res = requests.get("https://ipinfo.io/json", timeout=2)
                    if res.status_code == 200:
                        loc_data = res.json()
                        if 'loc' in loc_data:
                            lat, lon = loc_data['loc'].split(',')
                            st.session_state['ip_lat'] = float(lat)
                            st.session_state['ip_lon'] = float(lon)
                            st.session_state['ip_source'] = f"IP Fallback ({loc_data.get('city', 'Approximate')})"
                except Exception:
                    # Fallback to geocoder
                    try:
                        g = geocoder.ip('me')
                        if g.latlng:
                            st.session_state['ip_lat'] = float(g.latlng[0])
                            st.session_state['ip_lon'] = float(g.latlng[1])
                            st.session_state['ip_source'] = "IP Fallback (Approximate)"
                    except Exception:
                        pass
                        
            st.session_state['user_lat'] = st.session_state['ip_lat']
            st.session_state['user_lon'] = st.session_state['ip_lon']
            st.session_state['loc_source'] = st.session_state['ip_source']
            
        elif loc_method == "Manual Coordinates Override":
            st.sidebar.info("Enter custom coordinates below to simulate location movement.")
            m_lat = st.sidebar.number_input("Latitude", value=st.session_state['user_lat'], format="%.6f", key="man_lat")
            m_lon = st.sidebar.number_input("Longitude", value=st.session_state['user_lon'], format="%.6f", key="man_lon")
            st.session_state['user_lat'] = m_lat
            st.session_state['user_lon'] = m_lon
            st.session_state['loc_source'] = "Manual Coordinate Entry"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Settings")
    st.session_state['alert_radius'] = st.sidebar.slider(
        "Alert Radius (meters)", 
        min_value=10, 
        max_value=500, 
        value=st.session_state['alert_radius'], 
        step=10
    )
    
    # Audio Alert Toggle - Default to False
    st.session_state['enable_audio'] = st.sidebar.checkbox(
        "Enable Audio Warning Beeps", 
        value=st.session_state['enable_audio']
    )
    
    if st.session_state['enable_audio']:
        st.sidebar.caption("🔊 Sound is active. Beep alarm will trigger when approaching severe potholes.")
    else:
        st.sidebar.caption("🔇 Sound is muted. Check the box to enable alarms.")

    # Page routing configuration using st.Page
    pg_home = st.Page(home_page, title="Home & Overview", icon="🏠")
    pg_image = st.Page(image_page, title="Detect from Image", icon="📷")
    pg_video = st.Page(video_page, title="Detect from Video", icon="🎥")
    pg_camera = st.Page(camera_page, title="Real-time Camera", icon="📹")
    pg_map = st.Page(dashboard_page, title="Live Map & Alerts", icon="🗺️")

    pg = st.navigation([pg_home, pg_image, pg_video, pg_camera, pg_map])
    pg.run()

if __name__ == "__main__":
    main()
