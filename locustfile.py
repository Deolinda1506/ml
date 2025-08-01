import time
import random
import base64
import io
from locust import HttpUser, task, between
from PIL import Image
import numpy as np

def create_dummy_image(width=224, height=224):
    """Create a dummy image for testing"""
    # Create a random image
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

class GlaucomaDetectionUser(HttpUser):
    """Regular user for glaucoma detection"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when user starts"""
        print(f"User {self.user_id} started")
    
    @task(3)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/api/health")
    
    @task(2)
    def get_status(self):
        """Get system status"""
        self.client.get("/api/status")
    
    @task(5)
    def single_prediction(self):
        """Single image prediction"""
        # Create dummy image
        image_data = create_dummy_image()
        
        # Make prediction request
        response = self.client.post(
            "/api/predict",
            json={
                "image_data": image_data,
                "image_name": f"test_image_{random.randint(1000, 9999)}"
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Validate response
        if response.status_code == 200:
            result = response.json()
            if not result.get("success"):
                print(f"Prediction failed: {result}")
    
    @task(2)
    def bulk_prediction(self):
        """Bulk image prediction"""
        # Create multiple dummy images
        images = [create_dummy_image() for _ in range(random.randint(2, 5))]
        image_names = [f"bulk_test_{i}_{random.randint(1000, 9999)}" for i in range(len(images))]
        
        # Make bulk prediction request
        response = self.client.post(
            "/api/bulk-predict",
            json={
                "images": images,
                "image_names": image_names
            },
            headers={"Content-Type": "application/json"}
        )
        
        # Validate response
        if response.status_code == 200:
            result = response.json()
            if not result.get("success"):
                print(f"Bulk prediction failed: {result}")
    
    @task(1)
    def get_prediction_history(self):
        """Get prediction history"""
        self.client.get("/api/prediction-history?limit=10")
    
    @task(1)
    def get_visualizations(self):
        """Get data visualizations"""
        self.client.get("/api/visualizations")

class AdminUser(HttpUser):
    """Admin user for retraining and management"""
    wait_time = between(5, 10)
    weight = 1  # Less frequent than regular users
    
    def on_start(self):
        """Called when admin user starts"""
        print(f"Admin user {self.user_id} started")
    
    @task(2)
    def check_training_status(self):
        """Check training status"""
        self.client.get("/api/status")
    
    @task(1)
    def get_training_history(self):
        """Get training history"""
        self.client.get("/api/training-history")
    
    @task(1)
    def trigger_retraining(self):
        """Trigger model retraining"""
        response = self.client.post(
            "/api/retrain",
            json={
                "epochs": random.randint(10, 30),
                "batch_size": random.choice([16, 32, 64]),
                "learning_rate": random.uniform(0.0001, 0.01),
                "model_type": random.choice(["custom", "vgg16"])
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Retraining triggered: {result}")
    
    @task(1)
    def upload_training_data(self):
        """Upload training data (simulated)"""
        # Create dummy image files
        image_data = create_dummy_image()
        
        # Simulate file upload
        files = {
            'files': ('test_image.png', image_data, 'image/png')
        }
        data = {
            'class_labels': ['glaucoma']  # or 'normal'
        }
        
        # Note: This is a simplified version. Real file upload would use multipart/form-data
        response = self.client.post("/api/upload-data", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Data upload: {result}")

class SystemMonitor(HttpUser):
    """System monitoring user"""
    wait_time = between(10, 30)
    weight = 1
    
    def on_start(self):
        """Called when monitor starts"""
        print(f"System monitor {self.user_id} started")
    
    @task(3)
    def health_check(self):
        """Frequent health checks"""
        response = self.client.get("/api/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_system_status(self):
        """Get detailed system status"""
        response = self.client.get("/api/status")
        if response.status_code == 200:
            result = response.json()
            # Log system metrics
            metrics = result.get("system_metrics", {})
            print(f"System metrics: CPU={metrics.get('cpu_usage', 0)}%, "
                  f"Memory={metrics.get('memory_usage', 0)}%, "
                  f"Requests/min={metrics.get('requests_per_minute', 0)}")
    
    @task(1)
    def stress_test(self):
        """Stress test with multiple concurrent requests"""
        # Make multiple requests simultaneously
        responses = []
        for _ in range(5):
            response = self.client.get("/api/health")
            responses.append(response)
        
        # Check if all requests succeeded
        failed_requests = [r for r in responses if r.status_code != 200]
        if failed_requests:
            print(f"Stress test failed: {len(failed_requests)} requests failed")

# Custom events for monitoring
def on_request_success(request_type, name, response_time, response_length):
    """Called when a request succeeds"""
    print(f"SUCCESS: {request_type} {name} - {response_time}ms")

def on_request_failure(request_type, name, response_time, exception):
    """Called when a request fails"""
    print(f"FAILURE: {request_type} {name} - {response_time}ms - {exception}")

# Load test configuration
class LoadTestConfig:
    """Configuration for load testing"""
    
    @staticmethod
    def get_test_scenarios():
        """Get different test scenarios"""
        return {
            "light_load": {
                "users": 10,
                "spawn_rate": 2,
                "duration": "2m"
            },
            "medium_load": {
                "users": 50,
                "spawn_rate": 5,
                "duration": "5m"
            },
            "heavy_load": {
                "users": 100,
                "spawn_rate": 10,
                "duration": "10m"
            },
            "stress_test": {
                "users": 200,
                "spawn_rate": 20,
                "duration": "15m"
            }
        }
    
    @staticmethod
    def get_expected_performance():
        """Expected performance metrics"""
        return {
            "response_time_p95": 1000,  # 95th percentile should be under 1 second
            "response_time_p99": 2000,  # 99th percentile should be under 2 seconds
            "error_rate": 0.01,  # Error rate should be under 1%
            "requests_per_second": 50   # Should handle at least 50 RPS
        }

# Usage instructions
"""
To run load tests:

1. Start the FastAPI application:
   python src/app.py

2. Run Locust:
   locust -f locustfile.py --host=http://localhost:8000

3. Open browser and go to http://localhost:8089

4. Configure test parameters:
   - Number of users: 50
   - Spawn rate: 5 users/second
   - Host: http://localhost:8000

5. Start the test and monitor results

Expected results:
- Response time should be under 1 second for 95% of requests
- Error rate should be under 1%
- System should handle at least 50 requests per second
- Memory usage should remain stable
- CPU usage should scale with load
""" 