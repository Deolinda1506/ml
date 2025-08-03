# 👁️ Glaucoma Detection System

A deep learning-based system for detecting glaucoma from retinal images using convolutional neural networks (CNN).

## 🎯 Overview

This project provides a complete pipeline for glaucoma detection, including:
- **Model Training**: CNN-based image classification
- **Web Interface**: Streamlit app for easy interaction
- **API**: FastAPI backend for programmatic access
- **Data Management**: Upload and manage training data

## 📁 Project Structure

```
ml/
├── data/                    # Dataset directory
│   ├── train/              # Training images
│   │   ├── normal/         # Normal eye images
│   │   └── glaucoma/       # Glaucoma eye images
│   └── test/               # Test images
│       ├── normal/         # Normal eye images
│       └── glaucoma/       # Glaucoma eye images
├── src/                    # Source code
│   ├── model.py           # CNN model definition
│   ├── preprocessing.py   # Data preprocessing
│   ├── prediction.py      # Prediction functions
│   ├── retraining.py      # Model retraining
│   ├── app.py             # FastAPI application
│   └── database.py        # Database operations
├── UI/                    # User interface
│   └── streamlit_app.py   # Streamlit web app
├── notebook/              # Training and analysis
│   ├── training_script.py # Main training script
│   └── glaucoma.ipynb     # Jupyter notebook
├── models/                # Saved models (created after training)
├── static/                # Visualizations (created after training)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

   ```bash
# Clone the repository
   git clone <repository-url>
   cd ml

# Install dependencies
   pip install -r requirements.txt
   ```

### 2. Prepare Your Data

Organize your retinal images in the following structure:
```
data/
├── train/
│   ├── normal/     # Normal eye images (.jpg, .png)
│   └── glaucoma/   # Glaucoma eye images (.jpg, .png)
└── test/
    ├── normal/     # Normal eye images (.jpg, .png)
    └── glaucoma/   # Glaucoma eye images (.jpg, .png)
```

### 3. Train the Model

   ```bash
# Run the training script
python notebook/training_script.py
```

This will:
- Load and preprocess your data
- Train a CNN model
- Save the model to `models/best_model.h5`
- Generate visualizations in `static/`

### 4. Use the Web Interface

   ```bash
# Launch the Streamlit app
streamlit run UI/streamlit_app.py
```

### 5. Use the API

```bash
# Launch the FastAPI server
   python src/app.py
   ```

### 6. Load Testing

   ```bash
# Install Locust
   pip install locust

# Run load testing
   locust -f locustfile.py --host=http://localhost:8000

# Open browser to http://localhost:8089 for Locust web interface
```

## 🎯 Features

### 🔍 Single Image Prediction
- Upload individual retinal images
- Get instant predictions with confidence scores
- Visual confidence gauge
- Color-coded results

### 📊 Batch Processing
- Upload multiple images at once
- Batch analysis with progress tracking
- Download results as CSV
- Summary statistics and visualizations

### 📈 Dataset Analysis
- View training and test data distribution
- Class balance analysis
- Interactive charts and statistics

### 🤖 Model Training
- Train new models from scratch
- Retrain existing models
- View training history and metrics
- Model performance evaluation

### 📋 Data Management
- Upload new training images
- Organize data by class
- Add to existing dataset

## 🛠️ API Endpoints

### FastAPI Endpoints

- `GET /status` - System status and model information
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch image prediction
- `POST /upload` - Upload new training data
- `POST /retrain` - Retrain the model
- `GET /dataset_info` - Dataset statistics

### Example API Usage

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    result = response.json()
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']}")
```

## 📊 Model Architecture

The system uses a CNN with the following architecture:
- **Input**: 224x224x3 RGB images
- **Convolutional Layers**: 4 blocks with batch normalization
- **Pooling**: MaxPooling2D after each conv block
- **Dense Layers**: 256 → 128 → 2 (output classes)
- **Dropout**: 0.5 and 0.3 for regularization
- **Activation**: ReLU for hidden layers, Softmax for output

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## 🔧 Configuration

### Model Settings
- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Validation Split**: 20%

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip: Enabled
- Zoom: ±20%

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure you've run the training script first
   - Check that `models/best_model.h5` exists

2. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python path and module imports

3. **Memory issues**
   - Reduce batch size in training
   - Use smaller image size
   - Close other applications

4. **CUDA/GPU issues**
   - Install tensorflow-gpu for GPU support
   - Check CUDA compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical imaging community for datasets
- TensorFlow and Keras teams
- Streamlit and FastAPI developers

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**⚠️ Medical Disclaimer**: This system is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and clinical approval. 