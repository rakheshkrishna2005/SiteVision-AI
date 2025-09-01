# 🏗️ SiteVision-AI: Construction Site Intelligence Platform

A comprehensive AI-powered platform for construction site monitoring, progress tracking, and activity analysis using computer vision and geospatial visualization.

## 📋 Project Overview

SiteVision-AI consists of three interconnected modules that provide end-to-end construction site intelligence:

1. **Construction Activity Detection** - Real-time equipment and activity recognition using YOLOv8
2. **Construction Progress Analysis** - Semantic segmentation for progress measurement using U-Net
3. **Progress Map Visualization** - Interactive geospatial progress tracking using PyDeck



---

## 🔍 Module 1: Construction Activity Detection

### Description
Real-time construction equipment detection and activity analysis using YOLOv8 object detection. Monitors construction sites through video streams or uploaded footage to identify and track various types of construction equipment and activities.

**Key Features:**
- Real-time equipment detection (9 equipment classes)
- Activity stage classification
- Progress tracking and reporting
- Video processing capabilities
- Equipment activity mapping

### Tech Stack
- **AI Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Web Interface**: Streamlit
- **Computer Vision**: OpenCV
- **GPU Acceleration**: CUDA support

### Equipment Classes Detected
- 🚜 Paver
- 🚧 Compactor
- 🚜 Bulldozer
- 🚜 Excavator
- 🧱 Concrete Mixture
- 🏗️ Crane
- 🚛 Dump Truck
- 🔌 Utility Ducts
- 🚰 Concrete Pumps

### Screenshots
![Construction Activity App](https://github.com/rakheshkrishna2005/SiteVision-AI/tree/main/Construction%20Activity/app_screenshot.png)

### Usage
```bash
cd "Construction Activity"
streamlit run app.py
```

---

## 📊 Module 2: Construction Progress Analysis

### Description
Advanced construction progress measurement using U-Net semantic segmentation. Analyzes aerial imagery to measure construction progress, track road development, excavation status, and generate comprehensive progress reports.

**Key Features:**
- Semantic segmentation of construction elements
- Progress percentage calculations
- Historical progress tracking
- Automated report generation (PDF)
- MongoDB data persistence
- Interactive progress charts

### Tech Stack
- **AI Model**: U-Net (PyTorch)
- **Segmentation Classes**: Road, Excavated, Land, Cement, Background
- **Web Interface**: Streamlit
- **Data Storage**: MongoDB
- **Visualization**: Plotly, Matplotlib
- **Reporting**: ReportLab (PDF generation)

### Segmentation Classes
- 🛣️ **Road**: Constructed road surfaces
- ⚒️ **Excavated**: Areas under excavation
- 🌍 **Land**: Natural terrain
- 🧱 **Cement**: Cement/concrete structures
- ⚫ **Background**: Non-construction areas

### Screenshots
![Construction Progress App 1](https://github.com/rakheshkrishna2005/SiteVision-AI/tree/main/Construction%20Progress/app_screenshot1.jpg)
![Construction Progress App 2](https://github.com/rakheshkrishna2005/SiteVision-AI/tree/main/Construction%20Progress/app_screenshot2.jpg)

### Usage
```bash
cd "Construction Progress"
streamlit run app.py
```

---

## 🗺️ Module 3: Progress Map Visualization

### Description
Interactive geospatial visualization of construction progress across multiple sites using PyDeck and Mapbox. Provides a bird's-eye view of construction projects with real-time progress indicators and location-based analytics.

**Key Features:**
- Interactive 3D map visualization
- Multi-site progress tracking
- Real-time progress indicators
- Customizable map themes
- Location-based analytics
- Progress color coding

### Tech Stack
- **Mapping**: PyDeck (Deck.gl)
- **Map Provider**: Mapbox
- **Data Storage**: MongoDB
- **Web Interface**: Streamlit
- **Geospatial**: Coordinate-based visualization

### Map Features
- 🗺️ **Satellite Streets**: High-resolution satellite with street overlay
- 🛣️ **Streets**: Standard street map view
- 🛰️ **Satellite**: Pure satellite imagery
- 🎨 **Progress Indicators**: Color-coded construction status
- 📍 **Location Markers**: Site-specific information

### Screenshots
![Progress Map App](https://github.com/rakheshkrishna2005/SiteVision-AI/tree/main/Progress%20Map/app_screeenshot.jpg)

### Usage
```bash
cd "Progress Map"
streamlit run app.py
```

---

## 🔧 Technical Architecture

### Data Flow
```
Video/Image Input → YOLOv8 Detection → Activity Classification
                                                ↓
Aerial Imagery → U-Net Segmentation → Progress Measurement
                                                ↓
MongoDB Storage → PyDeck Visualization → Interactive Map
```

### Database Schema
```json
{
  "project_name": "string",
  "site_location": "string",
  "capture_date": "datetime",
  "drone_altitude": "float",
  "total_path_length": "float",
  "constructed_road": {
    "length": "float",
    "percentage": "float"
  },
  "excavation": {
    "length": "float",
    "percentage": "float"
  },
  "coordinates": {
    "latitude": "float",
    "longitude": "float"
  }
}
```
