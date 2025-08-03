from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
from .prediction import load_trained_model, predict_image, predict_batch
from .monitoring import monitor_request, get_metrics_collector

app = FastAPI(title="Glaucoma Detection API")
start_time = time.time()

# Load model at startup
MODEL_PATH = "models/best_model.h5"
CLASS_LABELS = {0: 'Normal', 1: 'Glaucoma'}  # Adapted for glaucoma detection

# Initialize model (will be loaded when available)
model = None

# Initialize metrics collector
metrics_collector = get_metrics_collector()

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """Load the trained model if it exists."""
    global model
    try:
        model = load_trained_model(MODEL_PATH)
        return True
    except FileNotFoundError:
        return False

@app.get("/status")
@monitor_request("status", "GET")
def get_status():
    uptime = time.time() - start_time
    model_loaded = model is not None
    
    # Record metrics
    metrics_collector.record_request("status", "GET", 0.1, 200)
    
    return {
        "status": "ok", 
        "uptime_seconds": int(uptime),
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
@monitor_request("predict", "POST")
async def predict(file: UploadFile = File(...)):
    start_time_request = time.time()
    
    if model is None:
        if not load_model():
            response_time = time.time() - start_time_request
            metrics_collector.record_request("predict", "POST", response_time, 500)
            return JSONResponse(
                content={"error": "Model not found. Please train the model first."}, 
                status_code=500
            )
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        response_time = time.time() - start_time_request
        metrics_collector.record_request("predict", "POST", response_time, 400)
        return JSONResponse(
            content={"error": "Invalid file type. Please upload PNG, JPG, or JPEG images."}, 
            status_code=400
        )
    
    contents = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    
    try:
        label, confidence = predict_image(model, temp_path, CLASS_LABELS)
        response_time = time.time() - start_time_request
        
        # Record successful prediction
        metrics_collector.record_request("predict", "POST", response_time, 200, prediction=label)
        
        return {
            "predicted_label": label, 
            "confidence": confidence,
            "filename": file.filename
        }
    except Exception as e:
        response_time = time.time() - start_time_request
        metrics_collector.record_request("predict", "POST", response_time, 500)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict_batch")
@monitor_request("predict_batch", "POST")
async def predict_batch_api(files: list[UploadFile] = File(...)):
    start_time_request = time.time()
    
    if model is None:
        if not load_model():
            response_time = time.time() - start_time_request
            metrics_collector.record_request("predict_batch", "POST", response_time, 500)
            return JSONResponse(
                content={"error": "Model not found. Please train the model first."}, 
                status_code=500
            )
    
    temp_paths = []
    results = []
    
    try:
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                results.append({
                    'image': file.filename, 
                    'error': 'Invalid file type. Please upload PNG, JPG, or JPEG images.'
                })
                continue
                
            contents = await file.read()
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(contents)
            temp_paths.append(temp_path)

        if temp_paths:
            batch_results = predict_batch(model, temp_paths, CLASS_LABELS)
            results.extend(batch_results)
            
        response_time = time.time() - start_time_request
        metrics_collector.record_request("predict_batch", "POST", response_time, 200)
            
    except Exception as e:
        response_time = time.time() - start_time_request
        metrics_collector.record_request("predict_batch", "POST", response_time, 500)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
    
    return {"results": results}

@app.post("/upload")
@monitor_request("upload", "POST")
async def upload_data(file: UploadFile = File(...), label: str = Form(...)):
    start_time_request = time.time()
    
    # Validate label
    if label.lower() not in ['normal', 'glaucoma']:
        response_time = time.time() - start_time_request
        metrics_collector.record_request("upload", "POST", response_time, 400)
        return JSONResponse(
            content={"error": "Invalid label. Use 'normal' or 'glaucoma'."}, 
            status_code=400
        )
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        response_time = time.time() - start_time_request
        metrics_collector.record_request("upload", "POST", response_time, 400)
        return JSONResponse(
            content={"error": "Invalid file type. Please upload PNG, JPG, or JPEG images."}, 
            status_code=400
        )
    
    try:
        # Save to database
        from .database import get_database
        db = get_database()
        
        # Save file to training directory
        train_dir = f"data/new_uploads/{label.lower()}"
        os.makedirs(train_dir, exist_ok=True)
        file_path = os.path.join(train_dir, file.filename)
        
        # Read and save file
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Save to database
        db.save_upload(
            filename=file.filename,
            file_path=file_path,
            label=label.lower(),
            file_size=len(file_content)
        )
        
        response_time = time.time() - start_time_request
        metrics_collector.record_request("upload", "POST", response_time, 200)
        
        return {
            "message": f"File {file.filename} uploaded to {train_dir}",
            "uploaded_path": file_path,
            "database_id": "saved"
        }
        
    except Exception as e:
        response_time = time.time() - start_time_request
        metrics_collector.record_request("upload", "POST", response_time, 500)
        return JSONResponse(
            content={"error": f"Upload failed: {str(e)}"}, 
            status_code=500
        )

@app.post("/retrain")
@monitor_request("retrain", "POST")
def retrain_model_api():
    start_time_request = time.time()
    
    try:
        import subprocess
        result = subprocess.run(["python", "src/retraining.py"], capture_output=True, text=True)
        
        # Reload the updated model
        global model
        if load_model():
            response_time = time.time() - start_time_request
            metrics_collector.record_request("retrain", "POST", response_time, 200)
            return {
                "message": "Retraining completed successfully.",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "model_reloaded": True
            }
        else:
            response_time = time.time() - start_time_request
            metrics_collector.record_request("retrain", "POST", response_time, 500)
            return {
                "message": "Retraining completed but model could not be reloaded.",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "model_reloaded": False
            }
    except Exception as e:
        response_time = time.time() - start_time_request
        metrics_collector.record_request("retrain", "POST", response_time, 500)
        return JSONResponse(
            content={"error": f"Retraining failed: {str(e)}"}, 
            status_code=500
        )

@app.get("/metrics")
def get_metrics():
    """Get system metrics and performance statistics."""
    return metrics_collector.get_metrics_summary()

@app.get("/metrics/prometheus")
def get_prometheus_metrics():
    """Get metrics in Prometheus format."""
    return metrics_collector.get_prometheus_metrics()

@app.get("/dataset_info")
def get_dataset_info():
    """Get information about the current dataset."""
    from .preprocessing import get_dataset_info
    
    train_info = get_dataset_info("data/train")
    test_info = get_dataset_info("data/test")
    
    return {
        "train_data": train_info,
        "test_data": test_info
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 