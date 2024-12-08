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

def main():
    st.markdown("<h1 style='text-align: center;'>🏗️ SiteVision AI", unsafe_allow_html=True)
    st.markdown("---")
    
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
                        
            # Add this line before the table generation
            total_progress = road_status['constructed_road_length'] + road_status['excavated_length']

            st.markdown(f"""
                <table style="width: 100%; background-color: #1c83ff1a; text-align: center;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px;">Activity</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Progress (meters)</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Percentage (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Road Construction</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{road_status['constructed_road_length']:.2f}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">
                                {(road_status['constructed_road_length'] / total_path_length * 100):.2f}%
                            </td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">Path Excavation</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{road_status['excavated_length']:.2f}</td>
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
            
            # Generate and display download button
            generate_download_link(export_df, f"{project_name}_path_analysis.csv")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
