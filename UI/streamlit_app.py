import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tempfile
import json
import requests
from datetime import datetime

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.prediction import load_trained_model, predict_image, predict_batch
from src.preprocessing import get_dataset_info
from src.model import build_model, get_data_generators, calculate_class_weights

# Page configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load the trained model."""
    try:
        model_path = "../models/best_model.h5"
        if os.path.exists(model_path):
            st.session_state.model = load_trained_model(model_path)
            st.session_state.model_loaded = True
            return True
        else:
            st.error("Model file not found. Please train the model first.")
            return False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Glaucoma Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Dataset Analysis", "🤖 Model Training", "📋 Upload Data"]
    )
    
    # Load model if not loaded
    if not st.session_state.model_loaded:
        if st.sidebar.button("🔄 Load Model", help="Load the trained glaucoma detection model"):
            with st.spinner("Loading model..."):
                load_model()
    
    # Display model status
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Loaded")
    else:
        st.sidebar.warning("⚠️ Model Not Loaded")
    
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
    
    # Try to get API status
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("API Status", "🟢 Online")
            with col2:
                st.metric("Uptime", f"{status_data.get('uptime_seconds', 0)}s")
            with col3:
                st.metric("Model Status", "✅ Loaded" if status_data.get('model_loaded', False) else "⚠️ Not Loaded")
            
            # Get metrics if available
            try:
                metrics_response = requests.get("http://localhost:8000/metrics", timeout=5)
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
    except:
        st.info("ℹ️ API not running. Start it with: `python src/app.py`")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **🔄 Load the Model**: Click 'Load Model' in the sidebar if not already loaded
    2. **📁 Upload an Image**: Go to 'Single Prediction' and upload a retinal image
    3. **📊 View Results**: Get instant prediction with confidence score
    4. **📈 Analyze Batch**: Use 'Batch Prediction' for multiple images
    """)
    
    # System information
    st.markdown("### ℹ️ System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Status", "✅ Ready" if st.session_state.model_loaded else "⚠️ Not Loaded")
    with col2:
        st.metric("API Status", "🟢 Online")
    with col3:
        st.metric("Version", "1.0.0")

def show_single_prediction():
    """Display single image prediction page."""
    st.markdown("## 🔍 Single Image Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first from the sidebar.")
        return
    
    # File upload section
    st.markdown("### 📁 Upload Retinal Image")
    st.markdown("Upload a retinal image for glaucoma detection analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a retinal image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal image for glaucoma detection"
    )
    
    if uploaded_file is not None:
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
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # Make prediction
                class_labels = {0: 'Normal', 1: 'Glaucoma'}
                label, confidence = predict_image(st.session_state.model, tmp_path, class_labels)
                
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
                st.error(f"❌ Error during prediction: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

def show_batch_prediction():
    """Display batch prediction page."""
    st.markdown("## 📊 Batch Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first from the sidebar.")
        return
    
    # File upload section
    st.markdown("### 📁 Upload Multiple Images")
    st.markdown("Upload multiple retinal images for batch analysis.")
    
    uploaded_files = st.file_uploader(
        "Choose multiple retinal image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple retinal images for batch analysis"
    )
    
    if uploaded_files:
        st.markdown(f"### 📋 Uploaded {len(uploaded_files)} Images")
        
        # Show uploaded files
        file_names = [f.name for f in uploaded_files]
        st.write("**Uploaded files:**", ", ".join(file_names))
        
        # Process images
        if st.button("🚀 Analyze All Images", help="Start batch analysis"):
            with st.spinner("🔄 Processing images..."):
                results = []
                temp_paths = []
                
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image = Image.open(uploaded_file)
                            image.save(tmp_file.name)
                            temp_paths.append(tmp_file.name)
                        
                        # Make prediction
                        class_labels = {0: 'Normal', 1: 'Glaucoma'}
                        label, confidence = predict_image(st.session_state.model, temp_paths[-1], class_labels)
                        
                        results.append({
                            'filename': uploaded_file.name,
                            'prediction': label,
                            'confidence': confidence,
                            'status': '✅ Normal' if label == 'Normal' else '⚠️ Glaucoma'
                        })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Analysis completed!")
                    
                    # Display results
                    df = pd.DataFrame(results)
                    
                    # Summary statistics
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
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"glaucoma_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.markdown("### 📊 Prediction Distribution")
                    fig = px.pie(df, names='prediction', title='Prediction Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during batch processing: {str(e)}")
                finally:
                    # Clean up temporary files
                    for temp_path in temp_paths:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

def show_dataset_analysis():
    """Display dataset analysis page."""
    st.markdown("## 📊 Dataset Analysis")
    
    st.markdown("### 📈 Dataset Overview")
    
    # Load dataset summary if available
    summary_path = "../static/dataset_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", summary['total_images'])
            with col2:
                st.metric("Image Shape", f"{summary['image_shape'][0]}x{summary['image_shape'][1]}")
            with col3:
                st.metric("Mean Pixel Value", f"{summary['mean_pixel_value']:.3f}")
            
            # Class distribution
            st.markdown("#### 📊 Class Distribution")
            class_data = summary['class_distribution']
            fig = go.Figure(data=[
                go.Bar(x=list(class_data.keys()), y=list(class_data.values()),
                      marker_color=['lightblue', 'lightcoral'])
            ])
            fig.update_layout(title="Class Distribution", xaxis_title="Classes", yaxis_title="Number of Images")
            st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading dataset summary: {str(e)}")
    
    # Display visualizations
    st.markdown("### 📸 Dataset Visualizations")
    
    viz_files = [
        ("Class Distribution", "class_distribution.png"),
        ("Sample Images", "sample_images.png"),
        ("Pixel Analysis", "pixel_analysis.png"),
        ("Image Characteristics", "image_characteristics.png")
    ]
    
    for title, filename in viz_files:
        file_path = f"../static/{filename}"
        if os.path.exists(file_path):
            st.markdown(f"#### {title}")
            st.image(file_path, use_column_width=True)
        else:
            st.info(f"📊 {title} visualization not found. Run the training script to generate it.")
    
    # Dataset statistics
    st.markdown("### 📋 Dataset Statistics")
    
    train_path = "/workspaces/ml/data/train"
    test_path = "/workspaces/ml/data/test"
    
    if os.path.exists(train_path):
        train_stats = {}
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                train_stats[class_name] = len(os.listdir(class_path))
        
        st.markdown("#### Training Data")
        for class_name, count in train_stats.items():
            st.write(f"**{class_name}**: {count} images")
    
    if os.path.exists(test_path):
        test_stats = {}
        for class_name in os.listdir(test_path):
            class_path = os.path.join(test_path, class_name)
            if os.path.isdir(class_path):
                test_stats[class_name] = len(os.listdir(class_path))
        
        st.markdown("#### Test Data")
        for class_name, count in test_stats.items():
            st.write(f"**{class_name}**: {count} images")

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
                    # Actually trigger retraining
                    import subprocess
                    import sys
                    import os
                    
                    # Change to the correct directory
                    current_dir = os.getcwd()
                    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
                    
                    # Run retraining script
                    result = subprocess.run(
                        [sys.executable, "src/retraining.py"], 
                        capture_output=True, 
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    # Change back to original directory
                    os.chdir(current_dir)
                    
                    if result.returncode == 0:
                        st.success("✅ Retraining completed successfully!")
                        st.info("Model has been updated with new data.")
                        
                        # Reload model
                        if st.session_state.model_loaded:
                            st.session_state.model_loaded = False
                            st.info("Please reload the model from the sidebar.")
                    else:
                        st.error(f"❌ Retraining failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    st.error("❌ Retraining timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Retraining failed: {str(e)}")
    
    # Model evaluation section
    st.markdown("### 📊 Model Evaluation")
    
    # Load evaluation results if available
    eval_path = "../static/evaluation_results.json"
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            
            # Key metrics
            st.markdown("#### 🎯 Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Accuracy", f"{eval_results['overall_metrics']['accuracy']:.4f}")
            with col2:
                st.metric("ROC AUC", f"{eval_results['roc_auc']:.4f}")
            with col3:
                st.metric("Average Precision", f"{eval_results['average_precision']:.4f}")
            with col4:
                st.metric("Macro F1-Score", f"{eval_results['overall_metrics']['macro_avg_f1']:.4f}")
            
            # Per-class metrics
            st.markdown("#### 📈 Per-Class Performance")
            per_class = eval_results['per_class_metrics']
            class_names = list(eval_results['classification_report'].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            
            class_metrics_df = pd.DataFrame({
                'Class': class_names,
                'Precision': per_class['precision'],
                'Recall': per_class['recall'],
                'F1-Score': per_class['f1_score']
            })
            
            st.dataframe(class_metrics_df, use_container_width=True)
            
            # Classification report
            st.markdown("#### 📋 Detailed Classification Report")
            report_df = pd.DataFrame(eval_results['classification_report']).T
            st.dataframe(report_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading evaluation results: {str(e)}")
    
    # Display evaluation visualizations
    st.markdown("### 📸 Evaluation Visualizations")
    
    eval_viz_files = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("ROC Curve", "roc_curve.png"),
        ("Precision-Recall Curve", "precision_recall_curve.png"),
        ("Training History", "training_history.png")
    ]
    
    for title, filename in eval_viz_files:
        file_path = f"../static/{filename}"
        if os.path.exists(file_path):
            st.markdown(f"#### {title}")
            st.image(file_path, use_column_width=True)
        else:
            st.info(f"📊 {title} visualization not found. Run the training script to generate it.")
    
    # Model information
    st.markdown("### 📋 Model Information")
    
    model_path = "../models/best_model.h5"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        st.info(f"📁 Model file exists: {model_path} ({file_size:.2f} MB)")
        
        # Check for training history
        history_path = "../models/history.json"
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                st.markdown("#### 📈 Training History")
                
                # Plot training metrics
                epochs = range(1, len(history['accuracy']) + 1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy'))
                fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy'))
                fig.update_layout(title='Model Accuracy Over Time', xaxis_title='Epoch', yaxis_title='Accuracy')
                st.plotly_chart(fig, use_container_width=True)
                
                # Final metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Training Accuracy", f"{history['accuracy'][-1]:.4f}")
                with col2:
                    st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                with col3:
                    st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
                with col4:
                    st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Error loading training history: {str(e)}")
    else:
        st.warning("⚠️ No model file found. Please train a model first.")

def show_upload_data():
    """Display data upload page."""
    st.markdown("## 📋 Upload Training Data")
    
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
            ["normal", "glaucoma"],
            help="Choose the correct classification for the uploaded images"
        )
        
        if st.button("📤 Upload to Training Data"):
            with st.spinner("🔄 Uploading images..."):
                try:
                    # Create directory structure
                    upload_dir = f"../data/new_uploads/{label}"
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Import database
                    import sys
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
                    from src.database import get_database
                    db = get_database()
                    
                    uploaded_count = 0
                    for uploaded_file in uploaded_files:
                        # Save file
                        file_path = os.path.join(upload_dir, uploaded_file.name)
                        file_content = uploaded_file.getbuffer()
                        
                        with open(file_path, "wb") as f:
                            f.write(file_content)
                        
                        # Save to database
                        db.save_upload(
                            filename=uploaded_file.name,
                            file_path=file_path,
                            label=label,
                            file_size=len(file_content)
                        )
                        
                        uploaded_count += 1
                    
                    st.success(f"✅ Successfully uploaded {uploaded_count} images to {upload_dir}")
                    st.info("Images are ready for retraining. Use the 'Retrain Model' button to include them in training.")
                    
                except Exception as e:
                    st.error(f"❌ Error uploading files: {str(e)}")

if __name__ == "__main__":
    main() 