"""
Database operations for Glaucoma Detection System
================================================

This module handles database operations for storing:
- Prediction results
- Model metadata
- Training history
- User uploads
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class GlaucomaDatabase:
    """Database manager for glaucoma detection system."""
    
    def __init__(self, db_path: str = "glaucoma_detection.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT,
                    processing_time REAL
                )
            ''')
            
            # Create model_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT NOT NULL,
                    model_version TEXT,
                    training_date DATETIME,
                    accuracy REAL,
                    validation_accuracy REAL,
                    epochs INTEGER,
                    batch_size INTEGER,
                    image_size TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create training_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create uploads table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.commit()
    
    def save_prediction(self, image_path: str, prediction: str, confidence: float, 
                       model_version: str = None, processing_time: float = None) -> int:
        """Save a prediction result to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (image_path, prediction, confidence, model_version, processing_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_path, prediction, confidence, model_version, processing_time))
            conn.commit()
            return cursor.lastrowid
    
    def get_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_model_metadata(self, model_path: str, model_version: str, 
                           accuracy: float, validation_accuracy: float,
                           epochs: int, batch_size: int, image_size: str) -> int:
        """Save model metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metadata 
                (model_path, model_version, training_date, accuracy, validation_accuracy, 
                 epochs, batch_size, image_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_path, model_version, datetime.now(), accuracy, validation_accuracy,
                  epochs, batch_size, image_size))
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_model_metadata(self) -> Optional[Dict]:
        """Get the latest model metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_metadata 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def save_training_history(self, model_version: str, history: Dict) -> None:
        """Save training history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for epoch in range(len(history.get('loss', []))):
                cursor.execute('''
                    INSERT INTO training_history 
                    (model_version, epoch, loss, accuracy, val_loss, val_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_version,
                    epoch + 1,
                    history.get('loss', [0])[epoch],
                    history.get('accuracy', [0])[epoch],
                    history.get('val_loss', [0])[epoch],
                    history.get('val_accuracy', [0])[epoch]
                ))
            
            conn.commit()
    
    def get_training_history(self, model_version: str = None) -> List[Dict]:
        """Get training history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if model_version:
                cursor.execute('''
                    SELECT * FROM training_history 
                    WHERE model_version = ?
                    ORDER BY epoch ASC
                ''', (model_version,))
            else:
                cursor.execute('''
                    SELECT * FROM training_history 
                    ORDER BY epoch ASC
                ''')
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def save_upload(self, filename: str, file_path: str, label: str, file_size: int) -> int:
        """Save upload record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO uploads (filename, file_path, label, file_size)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_path, label, file_size))
            conn.commit()
            return cursor.lastrowid
    
    def get_uploads(self, limit: int = 100) -> List[Dict]:
        """Get recent uploads."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM uploads 
                ORDER BY upload_date DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def update_upload_status(self, upload_id: int, status: str):
        """Update the status of an upload."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE uploads 
                SET status = ? 
                WHERE id = ?
            ''', (status, upload_id))
            conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Predictions by class
            cursor.execute('''
                SELECT prediction, COUNT(*) 
                FROM predictions 
                GROUP BY prediction
            ''')
            predictions_by_class = dict(cursor.fetchall())
            
            # Total uploads
            cursor.execute('SELECT COUNT(*) FROM uploads')
            total_uploads = cursor.fetchone()[0]
            
            # Uploads by label
            cursor.execute('''
                SELECT label, COUNT(*) 
                FROM uploads 
                GROUP BY label
            ''')
            uploads_by_label = dict(cursor.fetchall())
            
            # Model count
            cursor.execute('SELECT COUNT(*) FROM model_metadata')
            total_models = cursor.fetchone()[0]
            
            return {
                'total_predictions': total_predictions,
                'predictions_by_class': predictions_by_class,
                'total_uploads': total_uploads,
                'uploads_by_label': uploads_by_label,
                'total_models': total_models
            }
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data older than specified days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_predictions = cursor.rowcount
            
            # Delete old uploads
            cursor.execute('''
                DELETE FROM uploads 
                WHERE upload_date < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_uploads = cursor.rowcount
            
            conn.commit()
            return deleted_predictions + deleted_uploads

# Global database instance
db = GlaucomaDatabase()

def get_database() -> GlaucomaDatabase:
    """Get the global database instance."""
    return db 