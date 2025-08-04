"""
GLAUCOMA DETECTION - STREAMLIT WEB APP
======================================

A user-friendly web interface for glaucoma detection using deep learning.

Features:
- Single image prediction
- Batch image processing
- Dataset analysis
- Model training interface
- Data upload functionality

Usage:
    streamlit run UI/streamlit_app.py
"""

import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tempfile
from datetime import datetime

# API Configuration
API_URL = "https://ml-et3r.onrender.com"

# Page configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .normal-result {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .glaucoma-result {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .api-status-online {
        color: #28a745;
        font-weight: bold;
    }
    .api-status-offline {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'api_url' not in st.session_state:
    st.session_state.api_url = "https://ml-et3r.onrender.com"

def check_api_status(api_url):
    """Check if API is online."""
    try:
        response = requests.get(f"{api_url}/status", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Glaucoma Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar: Show API status
    with st.sidebar:
        st.header("🌐 API Configuration")
        
        # API URL selection
        api_url = st.selectbox(
            "Select API Endpoint",
            [
                "https://ml-et3r.onrender.com",
                "http://localhost:8000"
            ],
            index=0 if st.session_state.api_url == "https://ml-et3r.onrender.com" else 1
        )
        st.session_state.api_url = api_url
        
        st.markdown("---")
        
        # Check API status
        st.header("📊 API Status")
        is_online, status_data = check_api_status(api_url)
        
        if is_online:
            st.markdown('<p class="api-status-online">🟢 API Online</p>', unsafe_allow_html=True)
            st.write(f"**Uptime:** {status_data['uptime_seconds']} seconds")
            st.write(f"**Status:** {status_data['status']}")
            st.write(f"**Model Loaded:** {'✅ Yes' if status_data.get('model_loaded', False) else '❌ No'}")
        else:
            st.markdown('<p class="api-status-offline">🔴 API Offline</p>', unsafe_allow_html=True)
            st.error("Could not connect to backend.")
            
            if api_url == "https://ml-et3r.onrender.com":
                st.info("💡 **Tip:** Try switching to localhost:8000 if you have the API running locally")
                st.markdown("**To start local API:**")
                st.code("python src/app.py")
            else:
                st.info("💡 **Tip:** Make sure your local API is running on port 8000")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("## 🎯 Navigation")
        page = st.selectbox(
            "Choose a page",
            ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Dataset Analysis", "🤖 Model Training", "📋 Upload Data"]
        )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Single Prediction":
        show_single_prediction()
    elif page == "📊 Batch Prediction":
        show_batch_prediction()
    elif page == "📈 Dataset Analysis":
        show_dataset_analysis()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "📋 Upload Data":
        show_upload_data()

def show_home_page():
    """Display home page."""
    st.markdown("## 🏠 Welcome to Glaucoma Detection System")
    
    st.markdown("""
    This system uses advanced machine learning to detect glaucoma from retinal images.
    Upload retinal images to get instant predictions with confidence scores.
    """)
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Check API status
    is_online, status_data = check_api_status(st.session_state.api_url)
    
    if is_online:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("API Status", "🟢 Online")
        with col2:
            st.metric("Uptime", f"{status_data.get('uptime_seconds', 0)}s")
        with col3:
            st.metric("Model Status", "✅ Loaded" if status_data.get('model_loaded', False) else "⚠️ Not Loaded")
        
        # Get metrics if available
        try:
            metrics_response = requests.get(f"{st.session_state.api_url}/metrics", timeout=5)
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                if metrics:
                    st.markdown("### 📈 System Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.3f}s")
                    with col2:
                        st.metric("Total Requests", metrics.get('total_requests', 0))
                    with col3:
                        st.metric("Error Rate", f"{metrics.get('error_rate', 0):.1f}%")
                    with col4:
                        st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
        except:
            pass
    else:
        st.warning("⚠️ API is not responding")
        st.info(f"ℹ️ Current API URL: {st.session_state.api_url}")
        if st.session_state.api_url == "https://ml-et3r.onrender.com":
            st.info("💡 **To use local API:** Change the API endpoint in the sidebar to 'http://localhost:8000' and start the API with: `python src/app.py`")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **🌐 Connect to API**: Make sure the API is running (check sidebar status)
    2. **📁 Upload an Image**: Go to 'Single Prediction' and upload a retinal image
    3. **📊 View Results**: Get instant prediction with confidence score
    4. **📈 Analyze Batch**: Use 'Batch Prediction' for multiple images
    """)
    
    # System information
    st.markdown("### ℹ️ System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Status", "🟢 Online" if is_online else "🔴 Offline")
    with col2:
        st.metric("API URL", st.session_state.api_url.split("//")[-1])
    with col3:
        st.metric("Version", "1.0.0")

def show_single_prediction():
    """Display single image prediction page."""
    st.markdown("## 🔍 Single Image Prediction")
    
    # Check API status first
    is_online, _ = check_api_status(st.session_state.api_url)
    if not is_online:
        st.error("❌ API is not available. Please check the API status in the sidebar.")
        return
    
    # File upload section
    st.markdown("### 📁 Upload Retinal Image")
    st.markdown("Upload a retinal image for glaucoma detection analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a retinal image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal image for glaucoma detection"
    )
    
    if uploaded_file:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.markdown("#### 📋 Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
        
        with col2:
            st.markdown("### 🔍 Prediction Results")
            
            # Make prediction using API
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=30
                )
                result = response.json()
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    label = result.get('predicted_label', 'Unknown')
                    confidence = result.get('confidence', 0.0)
                    
                    # Display results
                    if label == 'Normal':
                        st.markdown(f"""
                        <div class="prediction-result normal-result">
                            <h3>✅ Normal - No Glaucoma Detected</h3>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p>No signs of glaucoma detected in this image. The optic nerve appears healthy.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result glaucoma-result">
                            <h3>⚠️ Glaucoma Detected</h3>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                            <p>Signs of glaucoma detected. Please consult an ophthalmologist for further evaluation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.markdown("#### 📊 Confidence Level")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Prediction Confidence"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def show_batch_prediction():
    """Display batch prediction page."""
    st.markdown("## 📊 Batch Prediction")
    
    # Check API status first
    is_online, _ = check_api_status(st.session_state.api_url)
    if not is_online:
        st.error("❌ API is not available. Please check the API status in the sidebar.")
        return
    
    # File upload section
    st.markdown("### 📁 Upload Multiple Images")
    st.markdown("Upload multiple retinal images for batch analysis.")
    
    batch_files = st.file_uploader(
        "Upload Multiple Retinal Images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="batch"
    )
    
    if batch_files:
        st.write(f"Uploaded {len(batch_files)} images")
        
        # Process images
        if st.button("🚀 Process Batch", key="batch_predict_btn"):
            with st.spinner("Processing batch predictions..."):
                try:
                    files = [("files", (f.name, f.getvalue())) for f in batch_files]
                    response = requests.post(f"{st.session_state.api_url}/predict_batch", files=files, timeout=60)
                    results = response.json()["results"]
                    
                    # Display results
                    for res in results:
                        if "error" in res:
                            st.error(f"{res.get('image', 'Unknown')}: {res['error']}")
                        else:
                            label = res.get('label', 'Unknown')
                            confidence = res.get('confidence', 0.0)
                            st.write(f"{res.get('image', 'Unknown')}: {label} (Confidence: {confidence:.2%})")
                    
                    # Summary statistics
                    if results:
                        df_data = []
                        for res in results:
                            if 'label' in res:
                                df_data.append({
                                    'filename': res.get('image', 'Unknown'),
                                    'prediction': res.get('label', 'Unknown'),
                                    'confidence': res.get('confidence', 0.0)
                                })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            st.markdown("### 📈 Summary Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Images", len(results))
                            with col2:
                                normal_count = len(df[df['prediction'] == 'Normal'])
                                st.metric("Normal", normal_count)
                            with col3:
                                glaucoma_count = len(df[df['prediction'] == 'Glaucoma'])
                                st.metric("Glaucoma", glaucoma_count)
                            with col4:
                                avg_confidence = df['confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                            
                            # Results table
                            st.markdown("### 📊 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Visualization
                            st.markdown("### 📊 Prediction Distribution")
                            fig = px.pie(df, names='prediction', title='Prediction Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

def show_dataset_analysis():
    """Display dataset analysis page."""
    st.markdown("## 📈 Dataset Analysis")
    
    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h4>Training Data</h4>', unsafe_allow_html=True)
        st.write("**Normal:** 100 images")
        st.write("**Glaucoma:** 100 images")
        st.write("**Total:** 200 images")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<h4>Test Data</h4>', unsafe_allow_html=True)
        st.write("**Normal:** 25 images")
        st.write("**Glaucoma:** 25 images")
        st.write("**Total:** 50 images")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Class distribution chart
    class_data = {'Normal': 100, 'Glaucoma': 100}
    fig = go.Figure(data=[go.Bar(x=list(class_data.keys()), y=list(class_data.values()),
                                marker_color=['lightblue', 'lightcoral'])])
    fig.update_layout(title="Class Distribution", xaxis_title="Classes", yaxis_title="Number of Images")
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Display model training page."""
    st.markdown("## 🤖 Model Training & Evaluation")
    
    # Training controls
    st.markdown("### 🚀 Training Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🆕 Train New Model")
        if st.button("🚀 Start Training"):
            with st.spinner("🔄 Training model..."):
                try:
                    # Actually trigger training
                    import subprocess
                    import sys
                    import os
                    
                    # Change to the correct directory
                    current_dir = os.getcwd()
                    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
                    
                    # Run training script
                    result = subprocess.run(
                        [sys.executable, "notebook/training_script.py"], 
                        capture_output=True, 
                        text=True,
                        timeout=600  # 10 minute timeout
                    )
                    
                    # Change back to original directory
                    os.chdir(current_dir)
                    
                    if result.returncode == 0:
                        st.success("✅ Training completed successfully!")
                        st.info("New model has been created and saved.")
                        
                        # Reload model
                        if st.session_state.model_loaded:
                            st.session_state.model_loaded = False
                            st.info("Please reload the model from the sidebar.")
                    else:
                        st.error(f"❌ Training failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    st.error("❌ Training timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        st.markdown("#### 🔄 Retrain Model")
        if st.button("🔄 Retrain Model"):
            with st.spinner("🔄 Retraining model..."):
                try:
                    response = requests.post(f"{st.session_state.api_url}/retrain", timeout=300)
                    result = response.json()
                    st.info(result.get("message", "Retraining triggered."))
                except Exception as e:
                    st.error(f"Could not start retraining: {e}")
    
    # Model Training History / Evaluation
    st.markdown("### 📊 Model Training History & Evaluation")
    
    try:
        with open("models/history.json", "r") as f:
            history = json.load(f)

        best_epoch = int(np.argmax(history["val_accuracy"]))  # Index of best val_accuracy

        epochs = np.arange(1, len(history["accuracy"]) + 1)
        best_epoch = best_epoch  # Adjust based on your best epoch

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Accuracy
        axs[0, 0].plot(epochs, history["accuracy"], label="Train Accuracy")
        axs[0, 0].plot(epochs, history["val_accuracy"], label="Val Accuracy")
        axs[0, 0].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
        axs[0, 0].set_title("Accuracy")
        axs[0, 0].legend()

        # Loss
        axs[0, 1].plot(epochs, history["loss"], label="Train Loss")
        axs[0, 1].plot(epochs, history["val_loss"], label="Val Loss")
        axs[0, 1].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
        axs[0, 1].set_title("Loss")
        axs[0, 1].legend()

        # AUC
        axs[1, 0].plot(epochs, history.get("auc", []), label="Train AUC")
        axs[1, 0].plot(epochs, history.get("val_auc", []), label="Val AUC")
        axs[1, 0].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
        axs[1, 0].set_title("AUC")
        axs[1, 0].legend()

        # Precision
        axs[1, 1].plot(epochs, history.get("precision", []), label="Train Precision")
        axs[1, 1].plot(epochs, history.get("val_precision", []), label="Val Precision")
        axs[1, 1].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
        axs[1, 1].set_title("Precision")
        axs[1, 1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Optionally, summarize best epoch metrics
        st.markdown(f"**Best Epoch (Early Stopping): {best_epoch+1}**")
        st.markdown(f"- Validation Accuracy: {history['val_accuracy'][best_epoch]:.3f}")
        st.markdown(f"- Validation Loss: {history['val_loss'][best_epoch]:.3f}")
        if 'val_auc' in history:
            st.markdown(f"- Validation AUC: {history['val_auc'][best_epoch]:.3f}")
        if 'val_precision' in history:
            st.markdown(f"- Validation Precision: {history['val_precision'][best_epoch]:.3f}")

    except Exception as e:
        st.warning("Training history not found or error reading history.json.")
        st.text(f"Error: {e}")

def show_upload_data():
    """Display data upload page."""
    st.markdown("## 📋 Upload Training Data")
    
    # Check API status first
    is_online, _ = check_api_status(st.session_state.api_url)
    if not is_online:
        st.error("❌ API is not available. Please check the API status in the sidebar.")
        return
    
    st.markdown("### 📁 Upload New Images")
    st.markdown("Upload new images to add to the training dataset.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose image files to add to training data",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images to add to the training dataset"
    )
    
    if uploaded_files:
        st.markdown(f"### 📋 Uploaded {len(uploaded_files)} Images")
        
        # Show uploaded files
        file_names = [f.name for f in uploaded_files]
        st.write("**Uploaded files:**", ", ".join(file_names))
        
        # Label selection
        label = st.selectbox(
            "Select the correct label for these images",
            ["normal", "glaucoma"]
        )
        
        if st.button("📤 Upload to Training Data"):
            with st.spinner("🔄 Uploading images..."):
                try:
                    # Upload each file
                    for uploaded_file in uploaded_files:
                        response = requests.post(
                            f"{st.session_state.api_url}/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                            data={"label": label},
                            timeout=30
                        )
                        if response.status_code == 200:
                            st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to upload: {uploaded_file.name}")
                    
                    st.info("Images are ready for retraining. Use the 'Retrain Model' button to include them in training.")
                    
                except Exception as e:
                    st.error(f"❌ Error uploading files: {str(e)}")

if __name__ == "__main__":
    main() 