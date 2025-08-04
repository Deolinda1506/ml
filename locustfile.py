"""
Locust Load Testing for Glaucoma Detection API
==============================================
Run this file with: 
    locust -f locustfile.py --host=https://ml-et3r.onrender.com
"""

import time
import random
import os
from locust import HttpUser, task, between

class BaseUser(HttpUser):
    wait_time = between(1, 5)
    model_path = os.environ.get("MODEL_PATH", "models/best_model.h5")

    def random_delay(self):
        time.sleep(random.uniform(0.1, 1.0))


class StatusCheckUser(BaseUser):
    @task
    def get_status(self):
        self.client.get("/status")
        self.random_delay()


class PredictUser(BaseUser):
    @task
    def post_predict(self):
        payload = {
            "patient_id": "12345",
            "age": 45,
            "gender": "M",
            "symptoms": ["blurry vision", "eye pain"]
        }
        self.client.post("/predict", json=payload)
        self.random_delay()


class ImagePredictUser(BaseUser):
    @task
    def post_image_predict(self):
        image_path = "test_images/test_image.jpg"
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                files = {
                    "file": ("test_image.jpg", image_file, "image/jpeg")
                }
                self.client.post("/predict/image", files=files)
        self.random_delay()


class StressTestUser(BaseUser):
    @task
    def stress_all_endpoints(self):
        # Status
        self.client.get("/status")

        # Predict
        payload = {
            "patient_id": "67890",
            "age": 60,
            "gender": "F",
            "symptoms": ["headache", "nausea"]
        }
        self.client.post("/predict", json=payload)

        # Image Predict
        image_path = "test_images/test_image.jpg"
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                files = {
                    "file": ("test_image.jpg", image_file, "image/jpeg")
                }
                self.client.post("/predict/image", files=files)

        self.random_delay()
