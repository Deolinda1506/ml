// Glaucoma Detection ML Pipeline - Frontend JavaScript

class GlaucomaDetectionApp {
    constructor() {
        this.initializeApp();
        this.setupEventListeners();
        this.loadDashboard();
        this.startPeriodicUpdates();
    }

    initializeApp() {
        console.log('Initializing Glaucoma Detection ML Pipeline...');
        this.initializeTooltips();
        this.setupCharts();
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    setupEventListeners() {
        // Single image prediction
        this.setupSingleImageUpload();
        this.setupBulkImageUpload();
        this.setupRetrainingForm();
        this.setupDataUploadForm();
        
        // Navigation
        this.setupNavigation();
        
        // Form submissions
        this.setupFormSubmissions();
    }

    setupSingleImageUpload() {
        const uploadArea = document.getElementById('single-upload-area');
        const fileInput = document.getElementById('single-image-input');
        const preview = document.getElementById('single-preview');
        const previewImg = document.getElementById('single-preview-img');
        const predictBtn = document.getElementById('predict-single-btn');
        const clearBtn = document.getElementById('clear-single-btn');

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                this.displayImagePreview(file, previewImg);
                preview.classList.remove('d-none');
                uploadArea.classList.add('d-none');
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.displayImagePreview(file, previewImg);
                preview.classList.remove('d-none');
                uploadArea.classList.add('d-none');
            }
        });

        // Predict button
        predictBtn.addEventListener('click', () => {
            this.predictSingleImage(fileInput.files[0]);
        });

        // Clear button
        clearBtn.addEventListener('click', () => {
            fileInput.value = '';
            preview.classList.add('d-none');
            uploadArea.classList.remove('d-none');
        });
    }

    setupBulkImageUpload() {
        const uploadArea = document.getElementById('bulk-upload-area');
        const fileInput = document.getElementById('bulk-image-input');
        const preview = document.getElementById('bulk-preview');
        const previewImages = document.getElementById('bulk-preview-images');
        const predictBtn = document.getElementById('predict-bulk-btn');
        const clearBtn = document.getElementById('clear-bulk-btn');

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                this.displayBulkImagePreview(files, previewImages);
                preview.classList.remove('d-none');
                uploadArea.classList.add('d-none');
            }
        });

        // Predict all button
        predictBtn.addEventListener('click', () => {
            this.predictBulkImages(Array.from(fileInput.files));
        });

        // Clear all button
        clearBtn.addEventListener('click', () => {
            fileInput.value = '';
            preview.classList.add('d-none');
            uploadArea.classList.remove('d-none');
        });
    }

    setupRetrainingForm() {
        const form = document.getElementById('retraining-form');
        const retrainBtn = document.getElementById('retrain-btn');
        const progressDiv = document.getElementById('training-progress');
        const progressBar = document.getElementById('training-progress-bar');
        const statusText = document.getElementById('training-status-text');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                epochs: parseInt(document.getElementById('epochs').value),
                batch_size: parseInt(document.getElementById('batch-size').value),
                learning_rate: parseFloat(document.getElementById('learning-rate').value),
                model_type: document.getElementById('model-type').value
            };

            retrainBtn.disabled = true;
            retrainBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';
            progressDiv.classList.remove('d-none');

            try {
                const response = await fetch('/api/retrain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                const result = await response.json();
                
                if (result.success) {
                    this.showAlert('Retraining started successfully!', 'success');
                    this.monitorTrainingProgress(progressBar, statusText);
                } else {
                    this.showAlert(result.message || 'Failed to start retraining', 'danger');
                }
            } catch (error) {
                this.showAlert('Error starting retraining: ' + error.message, 'danger');
            } finally {
                retrainBtn.disabled = false;
                retrainBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Retraining';
            }
        });
    }

    setupDataUploadForm() {
        const form = document.getElementById('upload-form');
        const statusDiv = document.getElementById('upload-status');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const classLabel = document.getElementById('class-select').value;
            const files = document.getElementById('training-images').files;

            if (!classLabel || files.length === 0) {
                this.showAlert('Please select a class and upload images', 'warning');
                return;
            }

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
                formData.append('class_labels', classLabel);
            }

            try {
                const response = await fetch('/api/upload-data', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.success) {
                    this.showAlert(`Successfully uploaded ${result.files.length} files`, 'success');
                    statusDiv.innerHTML = `<div class="alert alert-success">${result.message}</div>`;
                } else {
                    this.showAlert(result.message || 'Upload failed', 'danger');
                }
            } catch (error) {
                this.showAlert('Error uploading files: ' + error.message, 'danger');
            }
        });
    }

    setupNavigation() {
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupFormSubmissions() {
        // Prevent default form submissions and handle with JavaScript
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.id) {
                    e.preventDefault();
                }
            });
        });
    }

    displayImagePreview(file, previewElement) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewElement.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    displayBulkImagePreview(files, container) {
        container.innerHTML = '';
        files.forEach((file, index) => {
            const col = document.createElement('div');
            col.className = 'col-md-3 col-sm-4 col-6 mb-3';
            
            const reader = new FileReader();
            reader.onload = (e) => {
                col.innerHTML = `
                    <div class="card">
                        <img src="${e.target.result}" class="card-img-top" alt="Preview ${index + 1}">
                        <div class="card-body p-2">
                            <small class="text-muted">${file.name}</small>
                        </div>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
            container.appendChild(col);
        });
    }

    async predictSingleImage(file) {
        if (!file) return;

        this.showLoading('Processing image...');
        
        try {
            const base64 = await this.fileToBase64(file);
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_data: base64,
                    image_name: file.name
                })
            });

            const result = await response.json();
            this.hideLoading();
            
            if (result.success) {
                this.displaySinglePredictionResult(result);
                this.showAlert('Prediction completed successfully!', 'success');
            } else {
                this.showAlert('Prediction failed: ' + result.message, 'danger');
            }
        } catch (error) {
            this.hideLoading();
            this.showAlert('Error during prediction: ' + error.message, 'danger');
        }
    }

    async predictBulkImages(files) {
        if (files.length === 0) return;

        this.showLoading(`Processing ${files.length} images...`);
        
        try {
            const images = [];
            const imageNames = [];
            
            for (let file of files) {
                const base64 = await this.fileToBase64(file);
                images.push(base64);
                imageNames.push(file.name);
            }

            const response = await fetch('/api/bulk-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    images: images,
                    image_names: imageNames
                })
            });

            const result = await response.json();
            this.hideLoading();
            
            if (result.success) {
                this.displayBulkPredictionResults(result);
                this.showAlert(`Bulk prediction completed! Processed ${result.results.length} images`, 'success');
            } else {
                this.showAlert('Bulk prediction failed: ' + result.message, 'danger');
            }
        } catch (error) {
            this.hideLoading();
            this.showAlert('Error during bulk prediction: ' + error.message, 'danger');
        }
    }

    displaySinglePredictionResult(result) {
        const resultsDiv = document.getElementById('single-results');
        const prediction = result.prediction;
        const className = prediction.class === 1 ? 'glaucoma' : 'normal';
        const classLabel = prediction.class === 1 ? 'Glaucoma' : 'Normal';
        const confidenceClass = prediction.confidence > 0.8 ? 'high' : 
                               prediction.confidence > 0.6 ? 'medium' : 'low';

        resultsDiv.innerHTML = `
            <div class="prediction-result ${className}">
                <h5><i class="fas fa-${className === 'glaucoma' ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
                    ${classLabel}
                </h5>
                <p class="mb-2">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill ${confidenceClass}" style="width: ${prediction.confidence * 100}%"></div>
                </div>
                <small class="text-muted">Processing time: ${(result.processing_time * 1000).toFixed(0)}ms</small>
            </div>
        `;
    }

    displayBulkPredictionResults(result) {
        const tableBody = document.getElementById('bulk-results-table');
        const resultsDiv = document.getElementById('bulk-results');
        
        tableBody.innerHTML = '';
        
        result.results.forEach((item, index) => {
            if (item.success) {
                const className = item.prediction.class === 1 ? 'glaucoma' : 'normal';
                const classLabel = item.prediction.class === 1 ? 'Glaucoma' : 'Normal';
                const confidenceClass = item.prediction.confidence > 0.8 ? 'high' : 
                                      item.prediction.confidence > 0.6 ? 'medium' : 'low';

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><small>${item.image_name}</small></td>
                    <td><span class="badge bg-${className === 'glaucoma' ? 'danger' : 'success'}">${classLabel}</span></td>
                    <td>
                        <div class="d-flex align-items-center">
                            <span class="me-2">${(item.prediction.confidence * 100).toFixed(1)}%</span>
                            <div class="confidence-bar flex-grow-1" style="width: 60px;">
                                <div class="confidence-fill ${confidenceClass}" style="width: ${item.prediction.confidence * 100}%"></div>
                            </div>
                        </div>
                    </td>
                    <td><span class="badge bg-success">Success</span></td>
                `;
                tableBody.appendChild(row);
            } else {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><small>Image ${index + 1}</small></td>
                    <td>-</td>
                    <td>-</td>
                    <td><span class="badge bg-danger">Failed</span></td>
                `;
                tableBody.appendChild(row);
            }
        });
        
        resultsDiv.classList.remove('d-none');
    }

    async loadDashboard() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'operational') {
                this.updateDashboardMetrics(data);
                this.updateCharts(data);
            }
        } catch (error) {
            console.error('Error loading dashboard:', error);
        }
    }

    updateDashboardMetrics(data) {
        const stats = data.prediction_stats;
        
        document.getElementById('system-status').textContent = 'Operational';
        document.getElementById('total-predictions').textContent = stats.total_predictions || 0;
        document.getElementById('avg-confidence').textContent = 
            stats.avg_confidence ? `${(stats.avg_confidence * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('model-accuracy').textContent = 
            data.model_info?.accuracy ? `${(data.model_info.accuracy * 100).toFixed(1)}%` : 'N/A';
    }

    setupCharts() {
        // Prediction distribution chart
        const predictionCtx = document.getElementById('predictionChart');
        if (predictionCtx) {
            this.predictionChart = new Chart(predictionCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Normal', 'Glaucoma'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: ['#28a745', '#dc3545'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        // System metrics chart
        const metricsCtx = document.getElementById('metricsChart');
        if (metricsCtx) {
            this.metricsChart = new Chart(metricsCtx, {
                type: 'bar',
                data: {
                    labels: ['CPU', 'Memory', 'Disk'],
                    datasets: [{
                        label: 'Usage %',
                        data: [45, 68, 23],
                        backgroundColor: ['#007bff', '#28a745', '#ffc107'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    updateCharts(data) {
        // Update prediction chart
        if (this.predictionChart && data.prediction_stats.class_distribution) {
            const distribution = data.prediction_stats.class_distribution;
            this.predictionChart.data.datasets[0].data = [
                distribution.Normal || 0,
                distribution.Glaucoma || 0
            ];
            this.predictionChart.update();
        }

        // Update metrics chart
        if (this.metricsChart && data.system_metrics) {
            const metrics = data.system_metrics;
            this.metricsChart.data.datasets[0].data = [
                metrics.cpu_usage || 0,
                metrics.memory_usage || 0,
                metrics.disk_usage || 0
            ];
            this.metricsChart.update();
        }
    }

    async monitorTrainingProgress(progressBar, statusText) {
        const checkProgress = async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                const trainingStatus = data.training_status;
                
                if (trainingStatus.is_training) {
                    progressBar.style.width = `${trainingStatus.progress}%`;
                    statusText.textContent = trainingStatus.status;
                    
                    if (trainingStatus.progress < 100) {
                        setTimeout(checkProgress, 2000);
                    } else {
                        statusText.textContent = 'Training completed!';
                        this.showAlert('Model retraining completed successfully!', 'success');
                        this.loadDashboard();
                    }
                } else if (trainingStatus.error) {
                    statusText.textContent = 'Training failed';
                    this.showAlert('Training failed: ' + trainingStatus.error, 'danger');
                }
            } catch (error) {
                console.error('Error checking training progress:', error);
            }
        };
        
        checkProgress();
    }

    startPeriodicUpdates() {
        // Update dashboard every 30 seconds
        setInterval(() => {
            this.loadDashboard();
        }, 30000);
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    showLoading(message = 'Loading...') {
        document.getElementById('loading-text').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container');
        const alertId = 'alert-' + Date.now();
        
        const alertHtml = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        alertContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GlaucomaDetectionApp();
});

// Export for potential use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GlaucomaDetectionApp;
} 