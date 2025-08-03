# Database management for glaucoma detection system
import psycopg2
import psycopg2.extras
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class GlaucomaDatabase:
    """Database manager for glaucoma detection system."""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection."""
        if connection_string is None:
            # Default PostgreSQL connection string
            self.connection_string = "postgresql://glaucoma_db_aefp_user:Cti2rN9DbsyFwXukd7AzQjPvwf2pjIKR@dpg-d27mso8gjchc738e7vjg-a/glaucoma_db_aefp"
        else:
            self.connection_string = connection_string
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.connection_string)
    
    def init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT,
                    processing_time REAL
                )
            ''')
            
            # Create model_metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id SERIAL PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_version TEXT,
                    training_date TIMESTAMP,
                    accuracy REAL,
                    validation_accuracy REAL,
                    epochs INTEGER,
                    batch_size INTEGER,
                    image_size TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create training_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id SERIAL PRIMARY KEY,
                    model_version TEXT,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create uploads table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.commit()
    
    def save_prediction(self, image_path: str, prediction: str, confidence: float, 
                       model_version: str = None, processing_time: float = None) -> int:
        """Save a prediction result to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (image_path, prediction, confidence, model_version, processing_time)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            ''', (image_path, prediction, confidence, model_version, processing_time))
            conn.commit()
            return cursor.fetchone()[0]
    
    def get_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s
            ''', (limit,))
            return cursor.fetchall()
    
    def save_model_metadata(self, model_path: str, model_version: str, 
                           accuracy: float, validation_accuracy: float,
                           epochs: int, batch_size: int, image_size: str) -> int:
        """Save model metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metadata (model_path, model_version, training_date, accuracy, validation_accuracy, epochs, batch_size, image_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (model_path, model_version, datetime.now(), accuracy, validation_accuracy, epochs, batch_size, image_size))
            conn.commit()
            return cursor.fetchone()[0]
    
    def get_latest_model_metadata(self) -> Optional[Dict]:
        """Get the latest model metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('''
                SELECT * FROM model_metadata 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def save_training_history(self, model_version: str, history: Dict) -> None:
        """Save training history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Save each epoch's data
            for epoch in range(len(history.get('loss', []))):
                cursor.execute('''
                    INSERT INTO training_history (model_version, epoch, loss, accuracy, val_loss, val_accuracy)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (
                    model_version,
                    epoch + 1,
                    history.get('loss', [0])[epoch] if epoch < len(history.get('loss', [])) else 0,
                    history.get('accuracy', [0])[epoch] if epoch < len(history.get('accuracy', [])) else 0,
                    history.get('val_loss', [0])[epoch] if epoch < len(history.get('val_loss', [])) else 0,
                    history.get('val_accuracy', [0])[epoch] if epoch < len(history.get('val_accuracy', [])) else 0
                ))
            
            conn.commit()
    
    def get_training_history(self, model_version: str = None) -> List[Dict]:
        """Get training history."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if model_version:
                cursor.execute('''
                    SELECT * FROM training_history 
                    WHERE model_version = %s 
                    ORDER BY epoch
                ''', (model_version,))
            else:
                cursor.execute('''
                    SELECT * FROM training_history 
                    ORDER BY model_version, epoch
                ''')
            
            return cursor.fetchall()
    
    def save_upload(self, filename: str, file_path: str, label: str, file_size: int) -> int:
        """Save upload record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO uploads (filename, file_path, label, file_size)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            ''', (filename, file_path, label, file_size))
            conn.commit()
            return cursor.fetchone()[0]
    
    def get_uploads(self, limit: int = 100) -> List[Dict]:
        """Get recent uploads."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('''
                SELECT * FROM uploads 
                ORDER BY upload_date DESC 
                LIMIT %s
            ''', (limit,))
            return cursor.fetchall()
    
    def update_upload_status(self, upload_id: int, status: str):
        """Update the status of an upload."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE uploads 
                SET status = %s 
                WHERE id = %s
            ''', (status, upload_id))
            conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Get total uploads
            cursor.execute('SELECT COUNT(*) FROM uploads')
            total_uploads = cursor.fetchone()[0]
            
            # Get recent predictions (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            ''')
            recent_predictions = cursor.fetchone()[0]
            
            # Get prediction distribution
            cursor.execute('''
                SELECT prediction, COUNT(*) 
                FROM predictions 
                GROUP BY prediction
            ''')
            prediction_distribution = dict(cursor.fetchall())
            
            # Get upload distribution
            cursor.execute('''
                SELECT label, COUNT(*) 
                FROM uploads 
                GROUP BY label
            ''')
            upload_distribution = dict(cursor.fetchall())
            
            return {
                'total_predictions': total_predictions,
                'total_uploads': total_uploads,
                'recent_predictions_24h': recent_predictions,
                'prediction_distribution': prediction_distribution,
                'upload_distribution': upload_distribution
            }
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data older than specified days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE timestamp < NOW() - INTERVAL '%s days'
            ''', (days,))
            predictions_deleted = cursor.rowcount
            
            # Delete old uploads
            cursor.execute('''
                DELETE FROM uploads 
                WHERE upload_date < NOW() - INTERVAL '%s days'
            ''', (days,))
            uploads_deleted = cursor.rowcount
            
            conn.commit()
            return predictions_deleted + uploads_deleted

# Global database instance
_database_instance = None

def get_database() -> GlaucomaDatabase:
    """Get database instance (singleton pattern)."""
    global _database_instance
    if _database_instance is None:
        _database_instance = GlaucomaDatabase()
    return _database_instance 