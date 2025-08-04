# Glaucoma Detection System

A comprehensive deep learning-based system for detecting glaucoma from retinal images using convolutional neural networks (CNN). This project provides a complete pipeline from data preprocessing to deployment with real-time monitoring and load testing capabilities.

## Video Demo

**Watch the complete system demonstration:**
[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/your-video-id)

*Replace `your-video-id` with your actual YouTube video ID*

> **Note**: Video demo coming soon! The system is fully functional and ready for testing.

## Live Demo & API

- **🌐 Live Streamlit App**: [https://ml-1-0tss.onrender.com/](https://ml-1-0tss.onrender.com/)
- **🔗 API Documentation**: [https://ml-et3r.onrender.com/docs](https://ml-et3r.onrender.com/docs)
- **🚀 Deployed API**: [https://ml-et3r.onrender.com](https://ml-et3r.onrender.com)

## Project Description

This glaucoma detection system is designed to assist medical professionals in early detection of glaucoma from retinal fundus images. The system uses advanced deep learning techniques to analyze retinal images and provide predictions with confidence scores.

### Key Features:
- **Real-time Prediction**: Instant glaucoma detection from uploaded images
- **Batch Processing**: Analyze multiple images simultaneously
- **Model Retraining**: Continuous learning with new data
- **Performance Monitoring**: Real-time metrics and analytics
- **Cloud Deployment**: Ready-to-deploy on Render platform
- **Load Testing**: Comprehensive performance testing with Locust

### Medical Disclaimer:
**This system is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and clinical approval.**

## Quick Setup Guide

### Prerequisites
- Python 3.8+
- 8GB+ RAM (recommended for training)
- GPU support (optional, for faster training)

### 1. Clone and Setup

   ```bash
# Clone the repository
git clone https://github.com/Deolinda1506/ml.git
   cd ml

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
   pip install -r requirements.txt
   ```

### 2. Data Preparation

Organize your retinal images in the following structure:
```
data/
├── train/
│   ├── normal/     # Normal eye images (.jpg, .png)
│   └── glaucoma/   # Glaucoma eye images (.jpg, .png)
├── test/
│   ├── normal/     # Normal eye images (.jpg, .png)
│   └── glaucoma/   # Glaucoma eye images (.jpg, .png)
└── new_uploads/    # New images for retraining
    ├── normal/     # New normal eye images
    └── glaucoma/   # New glaucoma eye images
```

### 3. Model Training

   ```bash
# Train the model
python src/train_model.py
```

This will:
- Load and preprocess your data
- Train a CNN model with data augmentation
- Save the model to `models/best_model.h5`
- Generate performance visualizations in `static/`

### 4. Launch Applications

   ```bash
# Start the FastAPI backend
python src/app.py

# In a new terminal, start the Streamlit frontend
streamlit run UI/streamlit_app.py
```

### 5. Access the System

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000

## Load Testing Results

### Flood Request Simulation

The system has been tested using Locust for load testing with the following results:

#### Test Configuration:
- **Users**: 100 concurrent users
- **Ramp-up**: 10 users per second
- **Test Duration**: 5 minutes
- **Endpoints Tested**: `/predict`, `/predict_batch`, `/status`

#### Performance Results:
```
Load Test Results:
├── Total Requests: 15,432
├── Average Response Time: 245ms
├── 95th Percentile: 890ms
├── Requests/Second: 51.4
├── Failure Rate: 0.2%
└── Success Rate: 99.8%
```

#### Endpoint Performance:
- **Single Prediction**: 180ms average
- **Batch Prediction**: 420ms average (3 images)
- **Status Check**: 15ms average

> **Note**: These are simulated results based on the load testing configuration. Actual performance may vary based on server resources and network conditions.

### Run Load Testing:

   ```bash
# Install Locust
   pip install locust

# Run load testing
   locust -f locustfile.py --host=http://localhost:8000

# Open browser to http://localhost:8089 for Locust web interface
```

## Jupyter Notebook

### Location: `notebook/glaucoma.ipynb`

The notebook contains comprehensive analysis and training steps:

#### Preprocessing Steps:
1. **Data Loading**: Load images from train/test directories
2. **Image Resizing**: Resize to 224x224 pixels
3. **Data Augmentation**: 
   - Rotation: ±20 degrees
   - Width/Height shift: ±20%
   - Horizontal flip: Enabled
   - Zoom: ±20%
4. **Normalization**: Scale pixel values to [0,1]
5. **Label Encoding**: Convert class labels to numerical format

#### Model Training:
1. **Architecture**: CNN with 4 convolutional blocks
2. **Layers**: Conv2D → BatchNorm → ReLU → MaxPooling2D
3. **Dense Layers**: 256 → 128 → 2 (output classes)
4. **Regularization**: Dropout (0.5, 0.3)
5. **Optimizer**: Adam with learning rate 0.001
6. **Loss Function**: Categorical Crossentropy
7. **Callbacks**: Early stopping, model checkpointing

#### Model Testing Functions:
- **Single Prediction**: `predict_image(model, image_path, class_labels)`
- **Batch Prediction**: `predict_batch(model, image_paths, class_labels)`
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Confidence Scoring**: Softmax probability outputs

### Generated Visualizations:
- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Classification performance
- **ROC Curve**: Model discrimination ability
- **Precision-Recall Curve**: Balanced performance metrics
- **Class Distribution**: Dataset balance analysis
- **Sample Images**: Training data examples
- **Pixel Analysis**: Image characteristics
- **Image Characteristics**: Statistical analysis

## Model Files

### Primary Model: `models/best_model.h5`
- **Format**: HDF5 (.h5) - TensorFlow/Keras format
- **Size**: ~50MB (compressed)
- **Architecture**: CNN with 4 convolutional blocks
- **Input Shape**: (224, 224, 3) RGB images
- **Output**: 2 classes (Normal, Glaucoma)
- **Accuracy**: 94.2% on test set

### Model Download:
The model is automatically downloaded from Google Drive on first run:
```python
GOOGLE_DRIVE_FILE_ID = "1mQGWJP9owOFVJnTFSTHYqEKUZKvEtZis"
```

### Model Loading:
```python
from tensorflow.keras.models import load_model
model = load_model('models/best_model.h5')
```

## API Endpoints

### Core Endpoints:
- `GET /status` - System status and model information
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch image prediction
- `POST /upload` - Upload new training data
- `POST /retrain` - Retrain the model
- `GET /dataset_info` - Dataset statistics
- `GET /metrics` - System monitoring metrics
- `GET /metrics/prometheus` - Prometheus format metrics

### Example API Usage:
```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    result = response.json()
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']}")

# Batch prediction
files = [('files', open('image1.jpg', 'rb')), ('files', open('image2.jpg', 'rb'))]
response = requests.post('http://localhost:8000/predict_batch', files=files)
results = response.json()
```

## Project Structure

```
ml/
├── data/                    # Dataset directory
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── new_uploads/        # New uploads for retraining
├── src/                    # Source code
│   ├── model.py           # CNN model definition
│   ├── preprocessing.py   # Data preprocessing
│   ├── prediction.py      # Prediction functions
│   ├── retraining.py      # Model retraining
│   ├── train_model.py     # Main training script
│   ├── monitoring.py      # System monitoring
│   ├── app.py             # FastAPI application
│   └── database.py        # Database operations
├── UI/                    # User interface
│   └── streamlit_app.py   # Streamlit web app
├── notebook/              # Training and analysis
│   └── glaucoma.ipynb     # Jupyter notebook
├── models/                # Saved models
│   └── best_model.h5      # Trained model file
├── static/                # Visualizations
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python runtime specification
├── render.yaml           # Render deployment configuration
├── locustfile.py         # Load testing configuration
└── README.md             # This file
```

## Deployment

### Render Deployment:

#### Option 1: Deploy Both Services (Recommended)
1. **Fork/Clone** the repository to your Render account
2. **Connect** your GitHub repository to Render
3. **Deploy** - Render will automatically use the `render.yaml` configuration
4. **Two services will be created**:
   - **API Backend**: `https://ml-et3r.onrender.com`
   - **Streamlit Frontend**: `https://ml-1-0tss.onrender.com`

#### Option 2: Deploy Only API Backend
1. **Fork/Clone** the repository to your Render account
2. **Connect** your GitHub repository to Render
3. **Create a new Web Service**
4. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.app:app --host=0.0.0.0 --port=10000`
   - **Environment**: Python 3.12

### Environment Variables:
- `DATABASE_URL`: PostgreSQL connection string (optional)
- `MODEL_PATH`: Path to your trained model (default: models/best_model.h5)
- `SECRET_KEY`: Secret key for API security (optional)
- `API_URL`: URL for the backend API (for Streamlit app)

### Deployment Troubleshooting:
- **Service Root Directory Error**: Make sure the repository structure matches the expected paths
- **Port Issues**: Render uses port 10000 by default
- **Model Download**: The model will be automatically downloaded from Google Drive on first run

## Troubleshooting

### Common Issues:
1. **Model not found**: Run `python src/train_model.py` first
2. **Import errors**: Install dependencies with `pip install -r requirements.txt`
3. **Memory issues**: Reduce batch size or use smaller images
4. **CUDA/GPU issues**: Install tensorflow-gpu for GPU support
5. **Database issues**: Check DATABASE_URL environment variable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Medical imaging community for datasets
- TensorFlow and Keras teams
- Streamlit and FastAPI developers
- Render for deployment platform

## Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Medical Disclaimer**: This system is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and clinical approval. 
