import sqlite3
import json
from datetime import datetime
import os
from typing import Dict, List, Optional, Any

class DatabaseManager:
    def __init__(self, db_path="glaucoma_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                predicted_class TEXT,
                confidence REAL,
                actual_class TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                model_version TEXT
            )
        ''')
        
        # Training history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME,
                end_time DATETIME,
                epochs INTEGER,
                batch_size INTEGER,
                accuracy REAL,
                loss REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                status TEXT,
                model_path TEXT,
                training_data_size INTEGER,
                validation_data_size INTEGER
            )
        ''')
        
        # Data uploads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_name TEXT,
                file_size INTEGER,
                file_type TEXT,
                class_label TEXT,
                status TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        

        
        # Model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                model_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                is_active BOOLEAN DEFAULT FALSE,
                description TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, image_name: str, predicted_class: str, confidence: float, 
                      actual_class: Optional[str] = None, processing_time: float = 0.0, 
                      model_version: str = "latest"):
        """Log a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (image_name, predicted_class, confidence, actual_class, 
                                   processing_time, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_name, predicted_class, confidence, actual_class, processing_time, model_version))
        
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict]:
        """Get prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Class distribution
        cursor.execute('''
            SELECT predicted_class, COUNT(*) 
            FROM predictions 
            GROUP BY predicted_class
        ''')
        class_distribution = dict(cursor.fetchall())
        
        # Recent predictions (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE timestamp >= datetime('now', '-1 day')
        ''')
        recent_predictions = cursor.fetchone()[0]
        
        # Average processing time
        cursor.execute('SELECT AVG(processing_time) FROM predictions')
        avg_processing_time = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'avg_confidence': avg_confidence,
            'class_distribution': class_distribution,
            'recent_predictions': recent_predictions,
            'avg_processing_time': avg_processing_time
        }
    
    def log_training_start(self, epochs: int, batch_size: int, training_data_size: int, 
                          validation_data_size: int) -> int:
        """Log training start and return training ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_history (start_time, epochs, batch_size, training_data_size, 
                                        validation_data_size, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), epochs, batch_size, training_data_size, validation_data_size, 'started'))
        
        training_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return training_id
    
    def log_training_completion(self, training_id: int, accuracy: float, loss: float, 
                               precision: float, recall: float, f1_score: float, 
                               model_path: str):
        """Log training completion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_history 
            SET end_time = ?, accuracy = ?, loss = ?, precision = ?, recall = ?, 
                f1_score = ?, model_path = ?, status = 'completed'
            WHERE id = ?
        ''', (datetime.now(), accuracy, loss, precision, recall, f1_score, model_path, training_id))
        
        conn.commit()
        conn.close()
    
    def log_training_error(self, training_id: int, error_message: str):
        """Log training error"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_history 
            SET end_time = ?, status = ?
            WHERE id = ?
        ''', (datetime.now(), f'error: {error_message}', training_id))
        
        conn.commit()
        conn.close()
    
    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """Get training history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_history 
            ORDER BY start_time DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_last_training_time(self) -> Optional[datetime]:
        """Get last training time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT end_time FROM training_history 
            WHERE status = 'completed' 
            ORDER BY end_time DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None
    
    def log_data_upload(self, file_name: str, file_size: int, file_type: str, 
                       class_label: str, status: str = "uploaded"):
        """Log data upload"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_uploads (file_name, file_size, file_type, class_label, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_name, file_size, file_type, class_label, status))
        
        conn.commit()
        conn.close()
    
    def get_data_upload_history(self, limit: int = 50) -> List[Dict]:
        """Get data upload history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM data_uploads 
            ORDER BY upload_time DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    

    
    def save_model_version(self, version: str, model_path: str, accuracy: float, 
                          description: str = "", is_active: bool = True):
        """Save model version information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all other models if this one is active
        if is_active:
            cursor.execute('UPDATE model_versions SET is_active = FALSE')
        
        cursor.execute('''
            INSERT INTO model_versions (version, model_path, accuracy, description, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (version, model_path, accuracy, description, is_active))
        
        conn.commit()
        conn.close()
    
    def get_active_model_version(self) -> Optional[Dict]:
        """Get active model version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_versions 
            WHERE is_active = TRUE 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        return None
    


# Global database manager instance
db_manager = DatabaseManager()

def get_database_manager() -> DatabaseManager:
    """Get database manager instance"""
    return db_manager 