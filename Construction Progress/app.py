import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import csv
import io
from pymongo import MongoClient
from datetime import datetime
import json
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import plotly.io as pio
from dotenv import load_dotenv
import os

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")

# Constants (make sure these match your training constants)
CLASSES = ['_background_', 'Road', 'Excavated', 'Land', 'Cement']
COLOR_PALETTE = torch.tensor([
    [0, 0, 0],        # Background
    [27, 122, 0],     # Road
    [162, 115, 83],   # Excavated
    [110, 67, 38],    # Land
    [176, 176, 176]   # Cement
])

# Class Colors Dictionary (for calculations)
CLASS_COLORS = {
    "Road": (27, 122, 0),
    "Excavated": (162, 115, 83),
    "Background": (0, 0, 0),
    "Cement": (176, 176, 176),
    "Land": (110, 67, 38),
}

class ProjectDatabase:
    def __init__(self):
        # Connect to MongoDB
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['projectday']
        self.collection = self.db['projectdaydb']
        
        # Create indexes for better query performance
        self.collection.create_index([('project_name', 1)])
        self.collection.create_index([('capture_date', 1)])

    def save_progress(self, data):
        # Convert DataFrame row to dictionary and handle data types
        document = {
            'project_name': data['Project Name'],
            'site_location': data['Site Location'],
            'capture_date': datetime.strptime(data['Date'], '%Y-%m-%d'),
            'drone_altitude': float(data['Drone Altitude (m)']),
            'total_path_length': float(data['Total Path Length (m)']),
            'constructed_road': {
                'length': float(data['Road Construction (m)']),
                'percentage': float(data['Road Construction (%)'])
            },
            'excavation': {
                'length': float(data['Path Excavation (m)']),
                'percentage': float(data['Path Excavation (%)'])
            },
            'coordinates': {
                'latitude': float(data['Latitude']) if data['Latitude'] else None,
                'longitude': float(data['Longitude']) if data['Longitude'] else None
            },
            'created_at': datetime.utcnow()
        }
        
        # Insert into MongoDB
        self.collection.insert_one(document)

    def get_project_history(self, project_name):
        # Fetch all records for a project, sorted by date
        cursor = self.collection.find(
            {'project_name': project_name},
            {'_id': 0, 'capture_date': 1, 
             'constructed_road.percentage': 1, 
             'excavation.percentage': 1}
        ).sort('capture_date', 1)
        
        # Convert cursor to list of records
        history = list(cursor)
        
        # Format data for Streamlit chart
        if history:
            df = pd.DataFrame(history)
            df['Road Construction (%)'] = df['constructed_road'].apply(lambda x: x['percentage'])
            df['Path Excavation (%)'] = df['excavation'].apply(lambda x: x['percentage'])
            df = df[['capture_date', 'Road Construction (%)', 'Path Excavation (%)']]
            df = df.rename(columns={'capture_date': 'Date'})
            return df
        return None

    def get_latest_progress(self, project_name):
        # Get the most recent record for a project
        return self.collection.find_one(
            {'project_name': project_name},
            sort=[('capture_date', -1)]
        )
        
# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.dec4 = self._block(1024 + 512, 512)
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final(d1)

def calculate_longest_path(image, class_color):
    """
    Calculate the longest connected consecutive pixels for a specific class.
    """
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a mask for the specified class
    mask = np.all(image_rgb == class_color, axis=-1).astype(np.uint8)
    
    # Find the longest line of connected pixels in both directions
    horizontal_length = max([np.sum(row) for row in mask])
    vertical_length = max([np.sum(col) for col in mask.T])
    
    # Return the longer of the two
    return max(horizontal_length, vertical_length)

def compute_road_status(image, road_color, excavated_color, total_path_length, img_size=256, drone_altitude=60):
    """
    Compute the constructed road and excavated lengths within the total path length.
    """
    # Calculate pixel size (meters per pixel)
    pixel_size = drone_altitude / img_size
    
    # Compute the longest paths for Road and Excavated classes
    road_length_pixels = calculate_longest_path(image, road_color)
    excavated_length_pixels = calculate_longest_path(image, excavated_color)
    
    # Convert pixel lengths to real-world lengths
    road_length_meters = road_length_pixels * pixel_size
    excavated_length_meters = excavated_length_pixels * pixel_size
    
    # Scale road and excavated lengths to match total path length
    total_calculated_length = road_length_meters + excavated_length_meters
    if total_calculated_length > 0:
        scale_factor = total_path_length / total_calculated_length
        road_length_meters *= scale_factor
        excavated_length_meters *= scale_factor
    
    return {
        "constructed_road_length": road_length_meters,
        "excavated_length": excavated_length_meters
    }

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=len(CLASSES)).to(device)
    
    # Load the saved model state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def preprocess_image(image, img_size=256):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tensor

def create_color_mask(pred_mask):
    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_PALETTE):
        color_mask[pred_mask == i] = color.numpy()
    return color_mask

def create_cv2_color_mask(pred_mask):
    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_PALETTE):
        color_mask[pred_mask == i] = color.numpy()[::-1]  # Convert RGB to BGR
    return color_mask

def generate_download_link(export_data, filename):
    """
    Generate a download button for CSV data using Streamlit
    """
    # Convert the export data to a CSV
    output = io.StringIO()
    csv_writer = csv.writer(output)
    
    # Write headers
    csv_writer.writerow(export_data.columns)
    
    # Write data rows
    for _, row in export_data.iterrows():
        csv_writer.writerow(row)
    
    # Get the CSV content
    csv_content = output.getvalue()
    
    st.markdown(
    """
    <style>
    .st-emotion-cache-15hul6a {
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: fit-content !important;
    }
    """,
    unsafe_allow_html=True
    )
    
    # Create download button
    st.download_button(
        label="Download Analysis CSV",
        data=csv_content,
        file_name=filename,
        mime='text/csv'
    )
    
def create_legend():
    fig, ax = plt.subplots(figsize=(6, 1), facecolor='none')
    ax.set_facecolor('none')
    ax.axis('off')
    
    patches = []
    for class_name, color in CLASS_COLORS.items():
        if class_name != "Background":  # Exclude background from legend
            # Normalize color values to 0-1 range for matplotlib
            normalized_color = [c/255 for c in color]
            patch = mpatches.Patch(color=normalized_color, label=class_name)
            patches.append(patch)
    
    legend = plt.legend(handles=patches, loc='center', ncol=len(patches), frameon=False, facecolor='none', edgecolor='none')
    
    # Adjust legend text color if needed
    for text in legend.get_texts():
        text.set_color('white')  # or any color you prefer
    
    plt.tight_layout()
    st.pyplot(fig, facecolor='none', bbox_inches='tight', pad_inches=0)
    
def create_pdf_report(project_data, original_image, chart_fig):
    # Create buffer for PDF
    buffer = io.BytesIO()
    
    # Create PDF document with a border
    class BorderedDocTemplate(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            SimpleDocTemplate.__init__(self, *args, **kwargs)
            
        def beforePage(self):
            self.canv.setStrokeColorRGB(0.8, 0.8, 0.8)  # Light gray border
            self.canv.setLineWidth(3)
            self.canv.rect(
                20, 20,
                self.pagesize[0] - 40,
                self.pagesize[1] - 40
            )

    # Create document
    doc = BorderedDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=60,
        leftMargin=60,
        topMargin=60,
        bottomMargin=60
    )
    
    # List to hold PDF elements
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=28,
        spaceAfter=10,  # Reduced space to accommodate line
        alignment=1,  # Center alignment
        textColor=colors.HexColor('#1a237e')  # Dark blue
    )
    elements.append(Paragraph(f"SiteVision AI - Construction Progress Report", title_style))
    
    # Add a horizontal line
    elements.append(Spacer(1, 10))
    elements.append(Table([['']], colWidths=[doc.width], rowHeights=[1],
                         style=TableStyle([('LINEABOVE', (0, 0), (-1, -1), 1, colors.HexColor('#1a237e'))])))
    elements.append(Spacer(1, 20))
    
    # Create table data with more details
    table_data = [
        ["Project Details", "Values"],
        ["Project Name", project_data["Project Name"][0]],
        ["Site Location", project_data["Site Location"][0]],
        ["Date", project_data["Date"][0]],
        ["Drone Altitude", f"{project_data['Drone Altitude (m)'][0]} m"],
        ["Total Path Length", f"{project_data['Total Path Length (m)'][0]} m"],
        ["Road Construction", f"{project_data['Road Construction (m)'][0]} m ({project_data['Road Construction (%)'][0]}%)"],
        ["Path Excavation", f"{project_data['Path Excavation (m)'][0]} m ({project_data['Path Excavation (%)'][0]}%)"],
    ]
    
    # Enhanced table style with gradient-like effect
    table_style = TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),  # Dark blue header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        # Alternating row colors
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),  # Light gray
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),  # Dark gray text
        ('FONTNAME', (0, 1), (0, -1), 'Times-Bold'),  # Labels in bold
        ('FONTNAME', (1, 1), (1, -1), 'Times-Italic'),  # Values in italic
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),  # Light gray grid
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a237e')),  # Dark blue border
    ])
    
    # Update chart colors and font
    chart_fig.update_traces(
        marker_color='#1a237e',  # Dark blue
        selector=dict(name="Previous")
    )
    chart_fig.update_traces(
        marker_color='#90caf9',  # Light blue
        selector=dict(name="Current")
    )
    chart_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Times New Roman'),
        title=dict(
            text="Progress Comparison",
            font=dict(size=20, color='#1a237e', family='Times New Roman')
        )
    )
    
    # Create main content table
    # Process original image
    img_width = 15*15  # Square image
    orig_img_io = io.BytesIO()
    original_image.save(orig_img_io, format='PNG')
    orig_img_io.seek(0)
    orig_rl_img = RLImage(orig_img_io, width=img_width, height=img_width)  # Square dimensions
    
    # Create details table
    details_table = Table(table_data, colWidths=[2*inch, 3*inch])
    details_table.setStyle(table_style)
    
    # Arrange image and table side by side
    content_table = Table([
        [orig_rl_img, details_table]
    ], colWidths=[5*inch, 5.5*inch])  # Adjusted widths for A4
    content_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
    ]))
    
    elements.append(content_table)
    elements.append(Spacer(1, 20))
    
    # Add chart with increased height
    chart_bytes = pio.to_image(chart_fig, format='png', width=800, height=400)  # Increased height
    chart_io = io.BytesIO(chart_bytes)
    chart_rl_img = RLImage(chart_io, width=9*inch, height=4.5*inch)  # Increased height
    elements.append(chart_rl_img)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def add_pdf_download_button(project_data, original_image, _, chart_fig):
    try:
        pdf_buffer = create_pdf_report(project_data, original_image, chart_fig)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name=f"{project_data['Project Name'][0]}_analysis_report.pdf",
            mime='application/pdf'
        )
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def main():
    st.set_page_config(page_title="SiteVision Progress AI", page_icon="🏗️")

    st.markdown("<h1 style='text-align: center;'>🏗️ SiteVision AI", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; padding-bottom: 20px;">
        <a href=""><img src="https://img.shields.io/badge/Progress Tracking-Open-blue" style="margin-right: 10px;"></a>
        <a href=""><img src="https://img.shields.io/badge/Progress Map-Open-blue"></a>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("📋 Project Details")
    project_name = st.sidebar.text_input("Project Name")
    site_location = st.sidebar.text_input("Site Location")
    
    # Add drone altitude input
    drone_altitude = st.sidebar.number_input("Drone Altitude (meters)", min_value=1.0, value=60.0, step=1.0)
    
    # Total path length input with validation
    total_path_length = st.sidebar.number_input("Total Path Length (meters)", min_value=0.1, value=200.0, step=1.0)
    
    date = st.sidebar.date_input("Date")

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload Files", type=["jpg", "jpeg", "png"])
    
    # Add latitude and longitude inputs
    latitude_input = st.sidebar.text_input("Latitude (e.g., 12°59'56\"N)", placeholder="12°59'56\"N")
    longitude_input = st.sidebar.text_input("Longitude (e.g., 80°10'23\"E)", placeholder="80°10'23\"E")
    
    # Function to convert degrees, minutes, seconds to decimal degrees
    def dms_to_decimal(dms_str):
        try:
            # Remove the direction (N/S/E/W)
            direction = dms_str[-1]
            dms_str = dms_str[:-1]
            
            # Replace special characters and split
            dms_str = dms_str.replace('°', ' ').replace("'", ' ').replace('"', ' ')
            parts = dms_str.split()
            
            # Convert to float
            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            
            # Calculate decimal degrees
            decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
            
            # Apply sign based on direction
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
            
            return decimal_degrees
        except Exception as e:
            st.sidebar.error(f"Invalid coordinate format: {e}")
            return None
        
    # Convert inputs to decimal degrees
    latitude = dms_to_decimal(latitude_input) if latitude_input else None
    longitude = dms_to_decimal(longitude_input) if longitude_input else None
    
    if uploaded_file is not None:
        try:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            # Load model
            model_path = 'unet_model.pth'
            model, device = load_model(model_path)
            
            # Preprocess image
            input_tensor = preprocess_image(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_batch)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Create color visualization
            color_mask = create_color_mask(pred_mask)
            
            # Convert pred_mask to CV2 format for road length calculation
            cv2_mask = create_cv2_color_mask(pred_mask)
            
            # Compute road status
            road_status = compute_road_status(
                cv2_mask, 
                CLASS_COLORS["Road"], 
                CLASS_COLORS["Excavated"], 
                total_path_length,
                drone_altitude=drone_altitude
            )
            
            col1, col2 = st.columns(2)
            st.markdown(
            """
            <style>
                .stDownloadButton > button {
                    border-radius: 10px;
                }
            </style>
            """,
            unsafe_allow_html=True
            )
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(color_mask, caption="Segmented Image", use_column_width=True)
                
            st.markdown("""
            <style>
            /* Target the image elements within Streamlit image containers */
            .st-emotion-cache-1v0mbdj img {
                border-radius: 15px; /* Adjust the pixel value to control curve radius */
                object-fit: cover; /* Ensures image fills the container while maintaining aspect ratio */
            }
            
            /* Optional: Add a subtle shadow for depth (remove if not desired) */
            .st-emotion-cache-1v0mbdj img {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            create_legend()
                        
            # Add this line before the table generation
            total_progress = road_status['constructed_road_length'] + road_status['excavated_length']

            st.markdown(f"""
                <table style="width: 100%; background-color: #1c83ff1a; text-align: center;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px;">Activity</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Progress in meters</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Completed (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Road Construction</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{road_status['constructed_road_length']:.2f}m</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">
                                {(road_status['constructed_road_length'] / total_path_length * 100):.2f}%
                            </td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Path Excavation</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{road_status['excavated_length']:.2f}m</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">
                                {(road_status['excavated_length'] / total_path_length * 100):.2f}%
                            </td>
                        </tr>
                    </tbody>
                </table>
            """, unsafe_allow_html=True)

            # Prepare data for CSV export
            export_df = pd.DataFrame({
                "Project Name": [project_name],
                "Site Location": [site_location],
                "Date": [str(date)],
                "Drone Altitude (m)": [drone_altitude],
                "Total Path Length (m)": [total_path_length],
                "Road Construction (m)": [f"{road_status['constructed_road_length']:.2f}"],
                "Road Construction (%)": [f"{(road_status['constructed_road_length'] / total_path_length * 100):.2f}"],
                "Path Excavation (m)": [f"{road_status['excavated_length']:.2f}"],
                "Path Excavation (%)": [f"{(road_status['excavated_length'] / total_path_length * 100):.2f}"],
                "Latitude": [latitude],
                "Longitude": [longitude]
            })
            
            try:
                # Initialize database connection
                db = ProjectDatabase()
                
                previous_data = db.collection.find_one(
                    {
                        "project_name": project_name,
                        "site_location": site_location
                    },
                    sort=[('capture_date', -1)]  # Get the most recent record
                )

                if previous_data:
                    # Get current progress values
                    current_road = float(export_df['Road Construction (%)'].iloc[0])
                    current_excavation = float(export_df['Path Excavation (%)'].iloc[0])
                    
                    # Get previous progress values
                    prev_road = previous_data['constructed_road']['percentage']
                    prev_excavation = previous_data['excavation']['percentage']
                    
                    # Calculate progress differences
                    road_diff = current_road - prev_road
                    excavation_diff = current_excavation - prev_excavation
                    
                    # Check if road progress has decreased
                    if road_diff < 0:
                        st.error(f"Warning: Road Construction progress has decreased from {prev_road:.2f}% to {current_road:.2f}%")
                    
                    # Show success only if road progress has increased
                    if road_diff > 0:
                        st.success(f"""
                            Road Construction Progress Update:
                            - Progress Increased from {prev_road:.2f}% to {current_road:.2f}%
                            - Progresss Improvement Status: + {road_diff:.2f}%
                        """)
                    
                    # Create comparison bar chart
                    chart_data = pd.DataFrame({
                        'Category': ['Road Construction', 'Path Excavation'] * 2,
                        'Progress': [prev_road, prev_excavation, current_road, current_excavation],
                        'Type': ['Previous'] * 2 + ['Current'] * 2
                    })
                    
                    # Display bar chart
                    fig = px.bar(
                        chart_data, 
                        x='Category', 
                        y='Progress',
                        color='Type',
                        barmode='group',
                        title='Progress Comparison with Previous Site Images',
                        labels={'Progress': 'Percentage (%)'}
                    )
                    fig.update_layout(
                        legend=dict(
                            orientation='h',  # Horizontal legend
                            yanchor='bottom',
                            y=-0.35,  # Adjust this value to control space below plot
                            xanchor='center',
                            x=0.5  # Centers the legend
                        ),
                        margin=dict(b=100)  # Increases bottom margin to provide more space
                    )
                    st.plotly_chart(fig)
                    
                    # Add the new code here
                    if st.button("Generate PDF Report"):
                        add_pdf_download_button(export_df, image, color_mask, fig)
                    
                    st.markdown("""
                    <style>
                    /* Hide all legend titles */
                    .legendtitletext {
                        visibility: hidden !important;
                        height: 0 !important;
                        padding: 0 !important;
                        margin: 0 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                
                # Save new dataclea
                db.save_progress(export_df.iloc[0].to_dict())
                
            except Exception as db_error:
                st.error(f"Database error: {str(db_error)}")
            
            # Generate and display download button
            generate_download_link(export_df, f"{project_name}_path_analysis.csv")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()