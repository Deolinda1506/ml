"""
Locust Load Testing for Glaucoma Detection API
==============================================

This file simulates flood requests to test the API performance.
Run with: locust -f locustfile.py --host=http://localhost:8000
"""

import time
import random
from locust import HttpUser, task, between
import os
import base64

class GlaucomaDetectionUser(HttpUser):
    """Simulates users making requests to the glaucoma detection API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize test data on startup."""
        # Create test image data (base64 encoded small image)
        self.test_image_data = self.create_test_image()
    
    def create_test_image(self):
        """Create a simple test image for load testing."""
        # Create a minimal test image (1x1 pixel PNG)
        png_header = b'\x89PNG\r\n\x1a\n'
        png_data = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xe5\x0c\x1f\x0e\x1d\x0c\xc8\xc8\xc8\xc8\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\xea\x00\x00\x00\x00IEND\xaeB`\x82'
        return base64.b64encode(png_header + png_data).decode()
    
    @task(3)
    def test_single_prediction(self):
        """Test single image prediction endpoint."""
        try:
            # Create a test image file
            test_image = base64.b64decode(self.test_image_data)
        
            # Make prediction request
            with self.client.post(
                "/predict",
                files={"file": ("test_image.png", test_image, "image/png")},
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if "predicted_label" in result and "confidence" in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            response.failure(f"Request failed: {str(e)}")
    
    @task(1)
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        try:
            # Create multiple test images
            test_images = []
            for i in range(3):
                test_image = base64.b64decode(self.test_image_data)
                test_images.append(("files", ("test_image_{}.png".format(i), test_image, "image/png")))
            
            # Make batch prediction request
            with self.client.post(
                "/predict_batch",
                files=test_images,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if "results" in result and len(result["results"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid batch response format")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            response.failure(f"Batch request failed: {str(e)}")
    
    @task(1)
    def test_status_endpoint(self):
        """Test status endpoint."""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "status" in result and "model_loaded" in result:
                    response.success()
                else:
                    response.failure("Invalid status response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def test_dataset_info(self):
        """Test dataset info endpoint."""
        with self.client.get("/dataset_info", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "train_data" in result and "test_data" in result:
                    response.success()
                else:
                    response.failure("Invalid dataset info response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def test_upload_data(self):
        """Test data upload endpoint."""
        try:
            # Create test image for upload
            test_image = base64.b64decode(self.test_image_data)
            
            # Make upload request
            with self.client.post(
                "/upload",
                files={"file": ("upload_test.png", test_image, "image/png")},
                data={"label": "normal"},
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if "message" in result:
                        response.success()
                    else:
                        response.failure("Invalid upload response format")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            response.failure(f"Upload request failed: {str(e)}")

class GlaucomaDetectionLoadTest(HttpUser):
    """Heavy load testing for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Faster requests for load testing
    
    def on_start(self):
        """Initialize for load testing."""
        self.test_image_data = self.create_test_image()
    
    def create_test_image(self):
        """Create test image data."""
        png_header = b'\x89PNG\r\n\x1a\n'
        png_data = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xe5\x0c\x1f\x0e\x1d\x0c\xc8\xc8\xc8\xc8\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\xea\x00\x00\x00\x00IEND\xaeB`\x82'
        return base64.b64encode(png_header + png_data).decode()
    
    @task(10)
    def stress_test_prediction(self):
        """Stress test the prediction endpoint."""
        try:
            test_image = base64.b64decode(self.test_image_data)
            
            with self.client.post(
                "/predict",
                files={"file": ("stress_test.png", test_image, "image/png")},
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            response.failure(f"Stress test failed: {str(e)}")

# Configuration for different test scenarios
class Config:
    """Configuration for different load test scenarios."""
    
    @staticmethod
    def get_scenarios():
        """Get different test scenarios."""
        return {
            "normal_load": {
                "users": 10,
                "spawn_rate": 2,
                "duration": "5m"
            },
            "high_load": {
                "users": 50,
                "spawn_rate": 5,
                "duration": "10m"
            },
            "stress_test": {
                "users": 100,
                "spawn_rate": 10,
                "duration": "15m"
            }
        }

# Usage instructions
"""
To run load tests:

1. Start the API server:
   python src/app.py

2. Run normal load test:
   locust -f locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=5m

3. Run stress test:
   locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=15m

4. Open browser to http://localhost:8089 for Locust web interface

Expected Results:
- Response times under 2 seconds for normal load
- Response times under 5 seconds for high load
- Error rate under 5% for all scenarios
""" 