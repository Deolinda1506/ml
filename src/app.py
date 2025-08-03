from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
import gdown
from prediction import load_trained_model, predict_image, predict_batch
from monitoring import monitor_request, get_metrics_collector
from src.database import get_database  # ✅ Import your DB

app = FastAPI(title="Glaucoma Detection API")
start_time = time.time()

GOOGLE_DRIVE_FILE_ID = "1mQGWJP9owOFVJnTFSTHYqEKUZKvEtZis"
MODEL_PATH = "models/best_model.h5"
CLASS_LABELS = {0: 'Normal', 1: 'Glaucoma'}

model = None
metrics_collector = get_metrics_collector()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# On startup: download model + init DB
@app.on_event("startup")
def startup_event():
    try:
        download_model_from_drive()
        global model
        model = load_trained_model(MODEL_PATH)
        db = get_database()  # ✅ Initialize DB
        print("✅ Model and database initialized.")
    except Exception as e:
        print("❌ Startup failed:", e)

def download_model_from_drive():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            print("✅ Model downloaded.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

@app.get("/status")
@monitor_request("status", "GET")
def get_status():
    uptime = time.time() - start_time
    model_loaded = model is not None
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
        metrics_collector.record_request("predict", "POST", time.time() - start_time_request, 500)
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        metrics_collector.record_request("predict", "POST", time.time() - start_time_request, 400)
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        label, confidence = predict_image(model, temp_path, CLASS_LABELS)
        duration = time.time() - start_time_request
        metrics_collector.record_request("predict", "POST", duration, 200, prediction=label)

        # ✅ Save to database
        db = get_database()
        db.save_prediction(image_path=temp_path, prediction=label, confidence=confidence, processing_time=duration)

        return {
            "predicted_label": label,
            "confidence": confidence,
            "filename": file.filename
        }
    except Exception as e:
        metrics_collector.record_request("predict", "POST", time.time() - start_time_request, 500)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict_batch")
@monitor_request("predict_batch", "POST")
async def predict_batch_api(files: list[UploadFile] = File(...)):
    start_time_request = time.time()

    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    temp_paths = []
    results = []

    try:
        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                results.append({"image": file.filename, "error": "Invalid file type"})
                continue

            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            temp_paths.append(temp_path)

        if temp_paths:
            batch_results = predict_batch(model, temp_paths, CLASS_LABELS)
            results.extend(batch_results)

            # ✅ Save each to DB
            db = get_database()
            for res in batch_results:
                db.save_prediction(res["image"], res["label"], res["confidence"], processing_time=None)

        return {"results": results}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...), label: str = Form(...)):
    train_dir = f"data/new_uploads/{label}"
    os.makedirs(train_dir, exist_ok=True)
    file_path = os.path.join(train_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        db = get_database()
        file_size = os.path.getsize(file_path)
        db.save_upload(file.filename, file_path, label, file_size)
        print("✅ Upload logged.")
    except Exception as e:
        print("❌ Upload logging failed:", e)

    return {"message": f"File {file.filename} uploaded to {train_dir}"}

@app.post("/retrain")
def retrain_model_api():
    import subprocess
    result = subprocess.run(["python", "src/retrain.py"], capture_output=True, text=True)
    global model
    model = load_trained_model(MODEL_PATH)
    return {
        "message": "Retraining completed.",
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.get("/metrics")
def get_metrics():
    return metrics_collector.get_metrics_summary()

@app.get("/metrics/prometheus")
def get_prometheus_metrics():
    return metrics_collector.get_prometheus_metrics()

@app.get("/dataset_info")
def get_dataset_info():
    from preprocessing import get_dataset_info
    train_info = get_dataset_info("workspaces/ml/data/train")
    test_info = get_dataset_info("/workspaces/ml/data/test")
    return {
        "train_data": train_info,
        "test_data": test_info
    }

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
