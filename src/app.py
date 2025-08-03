from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
import requests
from prediction import load_trained_model, predict_image, predict_batch
from monitoring import monitor_request, get_metrics_collector

app = FastAPI(title="Glaucoma Detection API")
start_time = time.time()

# Model file info
MODEL_PATH = "models/best_model.h5"
CLASS_LABELS = {0: 'Normal', 1: 'Glaucoma'}

# Google Drive file ID for the model
GDRIVE_FILE_ID = "1mQGWJP9owOFVJnTFSTHYqEKUZKvEtZis"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Initialize model variable
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

def download_model_from_drive(dest_path: str):
    """Download the model from Google Drive if not present."""
    print(f"Downloading model from Google Drive to {dest_path} ...")
    session = requests.Session()

    URL = "https://docs.google.com/uc?export=download"

    response = session.get(URL, params={'id': GDRIVE_FILE_ID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': GDRIVE_FILE_ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)
    print("Model downloaded successfully.")

def get_confirm_token(response):
    """Helper to get confirm token for large files from Google Drive."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Helper to save streaming content to file."""
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def load_model():
    """Load the trained model if it exists; else download from drive and load."""
    global model

    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            download_model_from_drive(MODEL_PATH)
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False

    try:
        model = load_trained_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

@app.get("/status")
@monitor_request("status", "GET")
def get_status():
    uptime = time.time() - start_time
    model_loaded = model is not None
    
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

# Other routes (upload, retrain, metrics, etc.) remain unchanged, you can add them below.

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
