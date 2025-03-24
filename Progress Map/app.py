import streamlit as st
import pydeck as pdk
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import webbrowser
from dotenv import load_dotenv
import os

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")

class ProjectDatabase:
    def __init__(self):
        # Connect to MongoDB
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['projectday']
        self.collection = self.db['projectdaydb']

    def get_unique_locations(self):
        return self.collection.distinct('site_location')

    def get_latest_location_records(self):
        # Get the most recent record for each unique location
        unique_locations = self.get_unique_locations()
        latest_records = []

        for location in unique_locations:
            record = self.collection.find_one(
                {'site_location': location},
                sort=[('capture_date', -1)]
            )
            if record:
                latest_records.append(record)

        return latest_records

def format_road_progress(record):
    """
    Format road progress display based on the record
    Options:
    1. Percentage format: 70% completed
    2. Length format: 170m completed out of 200m
    """
    total_length = record.get('total_path_length', 0)
    constructed_length = record['constructed_road']['length']
    
    # Option 1: Percentage
    percentage = (constructed_length / total_length) * 100
    
    # Option 2: Length format
    length_format = f"{constructed_length:.1f} m out of {total_length:.0f} m"
    
    return f"{percentage:.1f}% ({length_format})"

def get_column_color(constructed_length, total_length):
    """
    Returns green if construction is complete, red otherwise
    """
    if constructed_length == total_length:
        return [0, 255, 0, 140]  # Green with alpha
    return [255, 0, 0, 140]  # Red with alpha

def main():
    st.set_page_config(
        page_title="Progress Map",
        page_icon="🏗️", 
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown("<h1 style='text-align: center;'>🗺️ Construction Progress Map</h1>", unsafe_allow_html=True)
    st.markdown("---")
            
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; padding-bottom: 20px;">
        <a href=""><img src="https://img.shields.io/badge/Progress Tracking-Open-blue" style="margin-right: 10px;"></a>
        <a href=""><img src="https://img.shields.io/badge/Progress Map-Open-blue"></a>
    </div>
    """, unsafe_allow_html=True)
    
    # Mapbox Style Dropdown
    mapbox_styles = {
        "Satellite Streets": "mapbox://styles/mapbox/satellite-streets-v11",
        "Streets": "mapbox://styles/mapbox/streets-v11",
        "Satellite": "mapbox://styles/mapbox/satellite-v9"
    }
    selected_style = st.sidebar.selectbox("Select Map Theme", list(mapbox_styles.keys()))

    # Define layout with margins
    left_margin, content, right_margin = st.columns([1, 6, 1])  # Adjust the ratio as needed

    with content:

        # Initialize database connection
        db = ProjectDatabase()
        
        # Get latest records for all locations
        latest_records = db.get_latest_location_records()
        
        # Create columns for project details
        project_cols = st.columns(3)
        
        project_details = []
        
        # Populate project columns with expandable sections
        for i, record in enumerate(latest_records[:3]):  # Limit to first 3 projects
            with project_cols[i]:
                with st.expander(f"{record['project_name']} - {record['site_location']}"):
                    # Extract and display project details
                    details = {
                        "Capture Date": record['capture_date'].strftime('%Y-%m-%d'),
                        "Drone Altitude": f"{record.get('drone_altitude', 'N/A')} m",
                        "Road Construction": format_road_progress(record),
                        "Path Excavation": f"{record['excavation']['length']:.1f} m ({record['excavation']['percentage']:.1f}%)"
                    }
                    st.table(pd.DataFrame.from_dict(details, orient='index', columns=['Value']))
                    
                    # Collect project details for map data
                    project_details.append(record)
            
        # Prepare map data
        map_data = []
        for record in project_details:
            # Collect coordinates for map if available
            coords = record.get('coordinates', {})
            if coords.get('latitude') and coords.get('longitude'):
                total_length = record.get('total_path_length', 0)
                constructed_length = record['constructed_road']['length']
                height = constructed_length / total_length * 100  # Normalize height to 0-100 range
                
                map_data.append({
                    'latitude': coords['latitude'],
                    'longitude': coords['longitude'],
                    'project_name': record['project_name'],
                    'site_location': record['site_location'],
                    'road_progress': format_road_progress(record),
                    'height': height,
                    'color': get_column_color(constructed_length, total_length)
                })
        
        # Create PyDeck map if we have location data
        if map_data:
            # Convert map data to DataFrame
            chart_data = pd.DataFrame(map_data)
            
            # Calculate map center
            center_lat = chart_data['latitude'].mean()
            center_lon = chart_data['longitude'].mean()

            # Define tooltip
            tooltip = {
                "html": """
                    <div style="background-color: aliceblue; color: black; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 4px rgba(0,0,0,0.25);">
                        <b>Project:</b> {project_name}<br/>
                        <b>Location:</b> {site_location}<br/>
                        <b>Road Progress:</b> {road_progress}<br/>
                    </div>
                """,
                "style": {
                    "position": "absolute",
                    "z-index": "10000",
                    "pointer-events": "none"
                }
            }
            
            # Create PyDeck map with 3D columns
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=65,
                height=600
            )

            column_layer = pdk.Layer(
                "ColumnLayer",
                data=chart_data,
                get_position=['longitude', 'latitude'],
                get_elevation='height',
                elevation_scale=100,
                radius=100,
                get_fill_color='color',
                pickable=True,
                auto_highlight=True,
                tooltip=tooltip
            )

            tile_layer = pdk.Layer(
                "TileLayer",
                data=None,
                opacity=0.5
            )

            deck = pdk.Deck(
                map_style=mapbox_styles[selected_style],  # Use the selected style
                initial_view_state=view_state,
                layers=[tile_layer, column_layer],
                tooltip=tooltip  # Add tooltip at deck level
            )

            st.pydeck_chart(deck)
        else:
            st.info("No location data available for projects.")

if __name__ == '__main__':
    main()