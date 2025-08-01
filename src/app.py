import os
import time
import base64
import io
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from preprocessing import ImagePreprocessor
from model import GlaucomaDetectionModel
from prediction import PredictionService
from retraining import get_retraining_service
from database import get_database_manager

# Initialize FastAPI app
app = FastAPI(
    title="Glaucoma Detection ML Pipeline",
    description="End-to-end machine learning pipeline for glaucoma detection from retinal fundus images",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize services
preprocessor = ImagePreprocessor()
prediction_service = PredictionService()
retraining_service = get_retraining_service()
db_manager = get_database_manager()

# Pydantic models for requests
class PredictionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    image_name: Optional[str] = None

class BulkPredictionRequest(BaseModel):
    images: List[str]  # List of base64 encoded images
    image_names: Optional[List[str]] = None

class RetrainingRequest(BaseModel):
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    model_type: str = "custom"

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting Glaucoma Detection ML Pipeline...")
    
    # Ensure model exists
    model_path = "models/glaucoma_model.h5"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
    else:
        print(f"Model loaded from {model_path}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Glaucoma Detection ML Pipeline...")

# Main dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

# API Endpoints

@app.post("/api/predict")
async def predict_single_image(request: PredictionRequest):
    """Predict single image"""
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image_data.split(',')[1] if ',' in request.image_data else request.image_data)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Preprocess image
        processed_image = preprocessor.load_and_preprocess_image_from_array(image_array)
        if processed_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Make prediction
        result = prediction_service.predict_single(processed_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log prediction
        image_name = request.image_name or f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db_manager.log_prediction(
            image_name=image_name,
            predicted_class="Glaucoma" if result['class'] == 1 else "Normal",
            confidence=result['confidence'],
            processing_time=processing_time
        )
        
        return {
            "success": True,
            "prediction": result,
            "processing_time": processing_time,
            "image_name": image_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/bulk-predict")
async def predict_multiple_images(request: BulkPredictionRequest):
    """Predict multiple images"""
    try:
        start_time = time.time()
        results = []
        
        for i, image_data in enumerate(request.images):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # Preprocess image
                processed_image = preprocessor.load_and_preprocess_image_from_array(image_array)
                if processed_image is None:
                    results.append({
                        "success": False,
                        "error": "Invalid image format",
                        "index": i
                    })
                    continue
                
                # Make prediction
                result = prediction_service.predict_single(processed_image)
                
                # Log prediction
                image_name = request.image_names[i] if request.image_names and i < len(request.image_names) else f"image_{i}"
                db_manager.log_prediction(
                    image_name=image_name,
                    predicted_class="Glaucoma" if result['class'] == 1 else "Normal",
                    confidence=result['confidence']
                )
                
                results.append({
                    "success": True,
                    "prediction": result,
                    "image_name": image_name,
                    "index": i
                })
                
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "index": i
                })
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "results": results,
            "total_time": total_time,
            "avg_time_per_image": total_time / len(request.images) if request.images else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk prediction error: {str(e)}")

@app.post("/api/upload-data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_labels: List[str] = Form(...)
):
    """Upload training data"""
    try:
        if len(files) != len(class_labels):
            raise HTTPException(status_code=400, detail="Number of files must match number of class labels")
        
        # Validate class labels
        valid_classes = ['glaucoma', 'normal']
        for label in class_labels:
            if label not in valid_classes:
                raise HTTPException(status_code=400, detail=f"Invalid class label: {label}. Must be one of {valid_classes}")
        
        # Upload files
        result = retraining_service.upload_training_data(files, class_labels)
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/api/retrain")
async def trigger_retraining(request: RetrainingRequest):
    """Trigger model retraining"""
    try:
        result = retraining_service.start_retraining(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            model_type=request.model_type
        )
        
        if result['success']:
            return result
        else:
            raise HTTPException(status_code=400, detail=result['message'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get system status and metrics"""
    try:
        # Get prediction stats
        prediction_stats = db_manager.get_prediction_stats()
        
        # Get training status
        training_status = retraining_service.get_training_status()
        
        # Get model info
        model_info = prediction_service.get_model_info()
        
        # Get system metrics (simulated)
        system_metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.4,
            "active_connections": 5,
            "requests_per_minute": 12.5
        }
        
        return {
            "status": "operational",
            "prediction_stats": prediction_stats,
            "training_status": training_status,
            "model_info": model_info,
            "system_metrics": system_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@app.get("/api/visualizations")
async def get_visualizations():
    """Get data visualizations"""
    try:
        # Analyze dataset
        analysis = preprocessor.analyze_dataset('../data/train')
        
        # Get prediction history for charts
        prediction_history = db_manager.get_prediction_history(100)
        
        # Prepare chart data
        class_distribution = analysis.get('class_distribution', {})
        file_formats = analysis.get('file_formats', {})
        
        # Prediction trends (last 7 days)
        recent_predictions = [p for p in prediction_history if p['timestamp']]
        
        return {
            "dataset_analysis": analysis,
            "class_distribution": class_distribution,
            "file_formats": file_formats,
            "prediction_trends": {
                "total": len(recent_predictions),
                "recent": len([p for p in recent_predictions if p['confidence'] > 0.8])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@app.get("/api/prediction-history")
async def get_prediction_history(limit: int = 50):
    """Get prediction history"""
    try:
        history = db_manager.get_prediction_history(limit)
        return {"predictions": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.get("/api/training-history")
async def get_training_history(limit: int = 10):
    """Get training history"""
    try:
        history = retraining_service.get_training_history(limit)
        return {"training_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training history error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 