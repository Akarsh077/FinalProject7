import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import os
import geocoder
from datetime import datetime
import requests
import json
import math
import folium
import streamlit.components.v1 as components
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
obj_names_path = os.path.join(BASE_DIR, 'utils', 'obj.names')

# Open the file using the absolute path

    # your code here



def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
    # Radius of the Earth in meters
    R = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c  # in meters

def get_location():
    """
    Get the current location using multiple methods for better accuracy.
    Falls back to IP-based location if more precise methods fail.
    """
    try:
        # Try to read from query parameters first
        try:
            lat = st.query_params.get('lat', None)
            lon = st.query_params.get('lon', None)
            if lat is not None and lon is not None:
                return float(lat), float(lon)
        except Exception as e:
            st.warning(f"Could not get location from query params: {e}")

        # Try to get location from browser (only works in Streamlit)
        if st.session_state.get('location') is None:
            # This will prompt the user for location permission in the browser
            st.session_state['location_requested'] = True

            # Add JavaScript to get location and set query parameters, reloading only if they changed
            st.markdown(
                """
                <script>
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        function(position) {
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            const urlParams = new URLSearchParams(window.location.search);
                            if (urlParams.get('lat') !== lat.toString() || urlParams.get('lon') !== lon.toString()) {
                                urlParams.set('lat', lat);
                                urlParams.set('lon', lon);
                                window.location.search = urlParams.toString();
                            }
                        },
                        function(error) {
                            console.error("Error getting location:", error);
                        }
                    );
                }
                </script>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"Could not get precise location: {e}")

    # Fallback to IP-based location
    try:
        # Try using ipinfo.io for more accurate IP-based location
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            if 'loc' in data:
                lat, lon = data['loc'].split(',')
                return float(lat), float(lon)
    except Exception as e:
        st.warning(f"Could not get location from ipinfo.io: {e}")

    # Final fallback to geocoder
    g = geocoder.ip('me')
    if g.latlng:
        return g.latlng[0], g.latlng[1]

    # If all methods fail, return a default location
    return 28.6139, 77.2090  # Default to New Delhi coordinates

def load_yolo_model():
    """
    Loads the YOLOv4 Tiny model and configures it.
    Uses CUDA backend if available, otherwise falls back to CPU.
    """
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(BASE_DIR.yolov4_tiny.weights)
    cfg_path = os.path.join(BASE_DIR, 'utils', 'yolov4_tiny.cfg')


    net = cv.dnn.readNet(weights_path, cfg_path)

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
    return model

def process_image(image):
    class_name = []
    with open(r'utils/obj.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    model = load_yolo_model()

    height, width, _ = image.shape
    image_area = height * width
    total_pothole_area = 0

    classes, scores, boxes = model.detect(image, 0.5, 0.4)

    # Get current location
    lat, lon = get_location()
    pothole_list = []

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        pothole_area = w * h
        total_pothole_area += pothole_area

        severity = "High" if (pothole_area / image_area) > 0.02 else "Medium" if (pothole_area / image_area) > 0.007 else "Low"

        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image, f"{label} ({severity})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add timestamp to the pothole data
        pothole_list.append([lat, lon, pothole_area, severity, timestamp])

    pothole_data = pd.DataFrame(pothole_list, columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"])
    return image, pothole_data

def process_video(video_path):
    class_name = []
    with open(obj_names_path, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    model = load_yolo_model()

    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to load video.")
        return None

    width = int(cap.get(3))
    height = int(cap.get(4))
    result = cv.VideoWriter('result.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (width, height))

    # Get current location
    lat, lon = get_location()
    pothole_list = []

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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

            # Simple tracking: check if this pothole is already detected (close in coordinates to any previously added)
            is_duplicate = False
            for item in pothole_list:
                prev_x, prev_y = item[5], item[6]
                if abs(x - prev_x) < 10 and abs(y - prev_y) < 10:
                    is_duplicate = True
                    break

            if not is_duplicate:
                pothole_list.append([lat, lon, pothole_area, severity, timestamp, x, y, w, h])

        result.write(frame)

    cap.release()
    result.release()
    cv.destroyAllWindows()

    # Create DataFrame from pothole list
    if pothole_list:
        pothole_data = pd.DataFrame(
            [item[:5] for item in pothole_list],
            columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"]
        )
        return pothole_data
    return None

def process_camera():
    class_name = []
    with open(obj_names_path, 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    model = load_yolo_model()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open camera.")
        return None

    width = int(cap.get(3))
    height = int(cap.get(4))

    # Get current location
    lat, lon = get_location()
    pothole_list = []

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    stframe = st.empty()

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

            # Simple tracking: check if this pothole is already detected (close in coordinates to any previously added)
            is_duplicate = False
            for item in pothole_list:
                prev_x, prev_y = item[5], item[6]
                if abs(x - prev_x) < 10 and abs(y - prev_y) < 10:
                    is_duplicate = True
                    break

            if not is_duplicate:
                pothole_list.append([lat, lon, pothole_area, severity, timestamp, x, y, w, h])

        stframe.image(frame, channels="BGR")

        if st.button("Stop Detection", key="stop_detection"):
            break

    cap.release()
    cv.destroyAllWindows()

    # Create DataFrame from pothole list
    if pothole_list:
        pothole_data = pd.DataFrame(
            [item[:5] for item in pothole_list],
            columns=["Latitude", "Longitude", "Pothole Area (pixels)", "Severity", "Timestamp"]
        )
        return pothole_data
    return None

def main():
    # Initialize session state for location
    if 'location' not in st.session_state:
        st.session_state['location'] = None
    if 'location_requested' not in st.session_state:
        st.session_state['location_requested'] = False

    st.title("Pothole Detection System")
    st.write("Upload an image or video to detect potholes using YOLOv4 Tiny.")

    # Display current location information
    lat, lon = get_location()
    st.sidebar.write(f"Current Location: {lat:.6f}, {lon:.6f}")
    st.sidebar.write("Note: For more accurate location, please allow location access in your browser.")

    option = st.radio("Select Input Type", ("Image", "Video", "Real-time Camera", "Live Map & Alert Dashboard"))

    if option == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            processed_image, pothole_data = process_image(image)
            # AREA CALCULATION
            st.table(pothole_data)
            height, width, _ = image.shape
            image_area = height * width
            total_pothole_area = pothole_data["Pothole Area (pixels)"].sum()
            st.write("Area % to maintain:",(total_pothole_area/image_area)*100 )
            st.image(processed_image, channels="BGR", caption="Processed Image")

            _, img_encoded = cv.imencode(".jpg", processed_image)
            st.download_button(
                label="Download Processed Image",
                data=img_encoded.tobytes(),
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )

            # Save CSV - MODIFIED TO APPEND DATA
            csv_path = "pothole_data.csv"

            # Check if file exists and append data
            if os.path.exists(csv_path):
                existing_data = pd.read_csv(csv_path)
                # Append new data
                updated_data = pd.concat([existing_data, pothole_data], ignore_index=True)
                updated_data.to_csv(csv_path, index=False)
                st.success(f"Added {len(pothole_data)} new pothole records to the database")
            else:
                # Create new file if it doesn't exist
                pothole_data.to_csv(csv_path, index=False)
                st.success(f"Created new database with {len(pothole_data)} pothole records")

            # Provide download option for the updated CSV
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download Pothole Data CSV",
                    data=f,
                    file_name="pothole_data.csv",
                    mime="text/csv"
                )

    elif option == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_video_path = "uploaded_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            if st.button("Run Detection", key="run_detection_video"):
                pothole_data = process_video(temp_video_path)

                # Save CSV - MODIFIED TO APPEND DATA FROM VIDEO
                if pothole_data is not None and not pothole_data.empty:
                    csv_path = "pothole_data.csv"

                    # Check if file exists and append data
                    if os.path.exists(csv_path):
                        existing_data = pd.read_csv(csv_path)
                        # Append new data
                        updated_data = pd.concat([existing_data, pothole_data], ignore_index=True)
                        updated_data.to_csv(csv_path, index=False)
                        st.success(f"Added {len(pothole_data)} new pothole records to the database")
                    else:
                        # Create new file if it doesn't exist
                        pothole_data.to_csv(csv_path, index=False)
                        st.success(f"Created new database with {len(pothole_data)} pothole records")

                st.video("result.avi")

                with open("result.avi", "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.avi",
                        mime="video/avi"
                    )

    elif option == "Real-time Camera":
        if st.button("Start Detection", key="start_detection_camera"):
            pothole_data = process_camera()

            # Save CSV - MODIFIED TO APPEND DATA FROM CAMERA
            if pothole_data is not None and not pothole_data.empty:
                csv_path = "pothole_data.csv"

                # Check if file exists and append data
                if os.path.exists(csv_path):
                    existing_data = pd.read_csv(csv_path)
                    # Append new data
                    updated_data = pd.concat([existing_data, pothole_data], ignore_index=True)
                    updated_data.to_csv(csv_path, index=False)
                    st.success(f"Added {len(pothole_data)} new pothole records to the database")
                else:
                    # Create new file if it doesn't exist
                    pothole_data.to_csv(csv_path, index=False)
                    st.success(f"Created new database with {len(pothole_data)} pothole records")

                # Provide download option for the updated CSV
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="Download Pothole Data CSV",
                        data=f,
                        file_name="pothole_data.csv",
                        mime="text/csv"
                    )

    elif option == "Live Map & Alert Dashboard":
        st.subheader("🗺️ Live Map & Proximity Alerts")

        # Load existing pothole data
        csv_path = "pothole_data.csv"
        if os.path.exists(csv_path):
            try:
                df_potholes = pd.read_csv(csv_path)
            except Exception as e:
                st.error(f"Error reading pothole data: {e}")
                df_potholes = pd.DataFrame()
        else:
            df_potholes = pd.DataFrame()
            st.info("No pothole database found yet. Detect potholes in Image/Video/Camera mode first.")

        # Get live location
        user_lat, user_lon = get_location()
        st.write(f"📍 **Your Current Location**: {user_lat:.6f}, {user_lon:.6f}")

        # Sidebar controls specifically for the dashboard
        st.sidebar.markdown("---")
        st.sidebar.header("Alert Settings")
        alert_radius = st.sidebar.slider("Alert Radius (meters)", min_value=10, max_value=500, value=50, step=10)
        enable_audio = st.sidebar.checkbox("Enable Audio Warning", value=True)

        # Center the map at the user's location
        m = folium.Map(location=[user_lat, user_lon], zoom_start=15)

        # Add a unique pin for user's live location (blue marker)
        folium.Marker(
            location=[user_lat, user_lon],
            popup="You are here",
            tooltip="Live Location",
            icon=folium.Icon(color="blue", icon="info-sign", prefix="glyphicon")
        ).add_to(m)

        # Draw the alert radius circle around the user
        folium.Circle(
            location=[user_lat, user_lon],
            radius=alert_radius,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.1,
            tooltip="Alert Zone"
        ).add_to(m)

        # Process potholes and check for alerts
        nearby_severe_potholes = []

        if not df_potholes.empty:
            severity_colors = {
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            }

            for idx, row in df_potholes.iterrows():
                p_lat = row['Latitude']
                p_lon = row['Longitude']
                p_sev = row.get('Severity', 'Low')
                p_area = row.get('Pothole Area (pixels)', 0)
                p_time = row.get('Timestamp', 'N/A')

                # Check distance
                distance = calculate_distance(user_lat, user_lon, p_lat, p_lon)

                # Populate lists for alert checking
                if p_sev == 'High' and distance <= alert_radius:
                    nearby_severe_potholes.append({
                        'index': idx + 1,
                        'distance': distance,
                        'lat': p_lat,
                        'lon': p_lon,
                        'area': p_area,
                        'time': p_time
                    })

                # Setup popup text
                popup_text = f"""
                <b>Pothole #{idx+1}</b><br>
                Severity: {p_sev}<br>
                Area: {p_area} px<br>
                Detected: {p_time}<br>
                Distance: {distance:.1f} m
                """

                color = severity_colors.get(p_sev, 'blue')
                folium.Marker(
                    location=[p_lat, p_lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Pothole: {p_sev} ({distance:.1f}m)",
                    icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon')
                ).add_to(m)

        # Display the map using Streamlit's HTML component
        components.html(m._repr_html_(), height=500)

        # Show alert warnings
        if nearby_severe_potholes:
            # Sort by distance
            nearby_severe_potholes = sorted(nearby_severe_potholes, key=lambda x: x['distance'])
            closest = nearby_severe_potholes[0]

            st.markdown(
                f"""
                <div style="background-color: #ff4b4b; padding: 15px; border-radius: 10px; border: 2px solid red; margin-bottom: 20px; animation: flash 1s infinite alternate;">
                    <h3 style="color: white; margin: 0; display: flex; align-items: center;">
                        🚨 DANGER: SEVERE POTHOLE AHEAD!
                    </h3>
                    <p style="color: white; font-size: 16px; margin: 10px 0 0 0;">
                        A very severe pothole is just <b>{closest['distance']:.1f} meters</b> away from your current location!
                    </p>
                </div>
                <style>
                @keyframes flash {{
                    from {{ box-shadow: 0 0 10px #ff4b4b; }}
                    to {{ box-shadow: 0 0 30px #ff0000; }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Play beep warning sound if audio is enabled
            if enable_audio:
                # Digital Watch Alarm sound
                components.html(
                    """
                    <div style="display:none;">
                        <audio id="alert-sound" autoplay loop>
                            <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
                        </audio>
                        <script>
                            // Handle potential auto-play restriction by checking user click state
                            document.addEventListener('click', function() {
                                var audio = document.getElementById('alert-sound');
                                if (audio) {
                                    audio.play().catch(e => console.log("Audio play failed: ", e));
                                }
                            }, {once: true});
                        </script>
                    </div>
                    """,
                    height=0
                )
                st.info("🔊 Warning audio is playing. (If you don't hear anything, click anywhere on the page to allow browser audio autoplay.)")

            # List all nearby severe potholes
            st.write("### ⚠️ Nearby Severe Potholes in Alert Zone:")
            for p in nearby_severe_potholes:
                st.warning(f"**Pothole #{p['index']}** is **{p['distance']:.1f} meters** away (Coordinates: {p['lat']:.5f}, {p['lon']:.5f})")
        else:
            st.success("✅ **Clear Road Ahead**: No severe potholes detected within the alert radius.")

        # Background Geolocation & Auto Rerun Script
        # We inject a JS script that watches position and only updates the query params if the location changes by more than 5 meters
        # This keeps the map updated without constant CPU spin or resetting controls when the vehicle is stationary.
        st.markdown(
            """
            <script>
            if (navigator.geolocation) {
                // Keep a single watch active
                if (!window.positionWatcherActive) {
                    window.positionWatcherActive = true;
                    console.log("Starting high-accuracy position watch...");

                    navigator.geolocation.watchPosition(
                        function(position) {
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            const urlParams = new URLSearchParams(window.location.search);
                            const prevLat = parseFloat(urlParams.get('lat') || '0');
                            const prevLon = parseFloat(urlParams.get('lon') || '0');

                            // Check if the change is significant (approx > 5 meters -> 0.00005 deg)
                            const latDiff = Math.abs(lat - prevLat);
                            const lonDiff = Math.abs(lon - prevLon);

                            if (latDiff > 0.00005 || lonDiff > 0.00005) {
                                console.log("Significant movement detected. Updating coordinates to: ", lat, lon);
                                urlParams.set('lat', lat);
                                urlParams.set('lon', lon);
                                window.location.search = urlParams.toString();
                            }
                        },
                        function(error) {
                            console.error("GPS Watch Position error:", error);
                        },
                        {
                            enableHighAccuracy: true,
                            timeout: 10000,
                            maximumAge: 0
                        }
                    );
                }
            } else {
                console.error("Geolocation not supported by this browser.");
            }
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()


