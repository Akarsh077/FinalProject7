import pandas as pd
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os


def visualize_potholes_on_map():
    # Read the CSV file containing pothole data
    try:
        df = pd.read_csv('pothole_data.csv')
        print(f"Loaded {len(df)} pothole records from CSV file")
    except FileNotFoundError:
        print("Error: pothole_data.csv file not found")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Check if the dataframe has the required columns
    required_columns = ['Latitude', 'Longitude', 'Severity']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        return
    
    # Create a map centered at the mean of coordinates
    center_lat = df['Latitude'].mean()
    center_lng = df['Longitude'].mean()
    pothole_map = folium.Map(location=[center_lat, center_lng], zoom_start=14)
    
    # Add a marker cluster to the map
    marker_cluster = MarkerCluster().add_to(pothole_map)
    
    # Define colors for different severity levels
    severity_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }
    
    # Add markers for each pothole
    for idx, row in df.iterrows():
        # Get the color based on severity
        color = severity_colors.get(row['Severity'], 'blue')
        
        # Create popup text with pothole information
        popup_text = f"""
        <b>Pothole #{idx+1}</b><br>
        Latitude: {row['Latitude']}<br>
        Longitude: {row['Longitude']}<br>
        Severity: {row['Severity']}
        """
        if 'Pothole Area (pixels)' in df.columns:
            popup_text += f"<br>Area: {row['Pothole Area (pixels)']} pixels"
        
        # Add marker to the cluster
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon='warning-sign', prefix='glyphicon')
        ).add_to(marker_cluster)
    
    # Save the map as an HTML file
    map_file = 'pothole_map.html'
    pothole_map.save(map_file)
    
    # Open the map in the default web browser
    map_path = os.path.abspath(map_file)
    print(f"Map saved to: {map_path}")
    webbrowser.open('file://' + map_path)

if __name__ == "__main__":
    visualize_potholes_on_map()
