import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TriathlonDatabase:
    """Helper class for database operations related to triathlon model predictions and feedback"""
    
    def __init__(self, db_path: str = 'triathlon_predictions.db'):
        self.db_path = db_path
    
    def save_prediction(self, image_path: str, predicted_class: str, confidence: float) -> int:
        """Save a prediction to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (image_path, predicted_class, confidence, created_at)
                VALUES (?, ?, ?, ?)
            """, (image_path, predicted_class, confidence, datetime.now()))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Saved prediction {prediction_id}: {predicted_class} ({confidence:.3f})")
            return prediction_id
            
        except sqlite3.Error as e:
            logger.error(f"Error saving prediction: {e}")
            return -1
    
    def save_correction(self, prediction_id: int, actual_class: str) -> bool:
        """Save a user correction for a prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE predictions 
                SET actual_class = ?, corrected_at = ?
                WHERE id = ?
            """, (actual_class, datetime.now(), prediction_id))
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                logger.info(f"Saved correction for prediction {prediction_id}: {actual_class}")
                return True
            else:
                logger.warning(f"No prediction found with id {prediction_id}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error saving correction: {e}")
            return False
    
    def save_correction_by_image(self, image_path: str, predicted_class: str, actual_class: str) -> bool:
        """Save a correction by finding the most recent prediction for an image"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find the most recent prediction for this image
            cursor.execute("""
                SELECT id FROM predictions 
                WHERE image_path = ? AND predicted_class = ?
                ORDER BY created_at DESC 
                LIMIT 1
            """, (image_path, predicted_class))
            
            result = cursor.fetchone()
            
            if result:
                prediction_id = result[0]
                cursor.execute("""
                    UPDATE predictions 
                    SET actual_class = ?, corrected_at = ?
                    WHERE id = ?
                """, (actual_class, datetime.now(), prediction_id))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Saved correction for {image_path}: {actual_class}")
                return True
            else:
                conn.close()
                logger.warning(f"No matching prediction found for {image_path}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error saving correction by image: {e}")
            return False
    
    def get_corrections_count(self, since_date: Optional[datetime] = None) -> int:
        """Get count of corrections, optionally since a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if since_date:
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE actual_class IS NOT NULL AND corrected_at > ?
                """, (since_date,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE actual_class IS NOT NULL
                """)
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except sqlite3.Error as e:
            logger.error(f"Error getting corrections count: {e}")
            return 0
    
    def get_recent_corrections(self, limit: int = 50) -> List[Tuple]:
        """Get recent corrections for retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT image_path, predicted_class, actual_class, confidence, corrected_at
                FROM predictions 
                WHERE actual_class IS NOT NULL
                ORDER BY corrected_at DESC
                LIMIT ?
            """, (limit,))
            
            corrections = cursor.fetchall()
            conn.close()
            
            return corrections
            
        except sqlite3.Error as e:
            logger.error(f"Error getting recent corrections: {e}")
            return []
    
    def get_model_accuracy_trend(self) -> List[Tuple]:
        """Get model accuracy over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT version, accuracy, training_date
                FROM model_versions
                ORDER BY version ASC
            """)
            
            trend = cursor.fetchall()
            conn.close()
            
            return trend
            
        except sqlite3.Error as e:
            logger.error(f"Error getting accuracy trend: {e}")
            return []
    
    def get_prediction_stats(self) -> dict:
        """Get overall prediction statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Total corrections
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_class IS NOT NULL")
            total_corrections = cursor.fetchone()[0]
            
            # Accuracy of recent predictions
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_class = actual_class THEN 1 ELSE 0 END) as correct
                FROM predictions 
                WHERE actual_class IS NOT NULL
            """)
            result = cursor.fetchone()
            
            if result[0] > 0:
                accuracy = result[1] / result[0]
            else:
                accuracy = 0.0
            
            # Class distribution
            cursor.execute("""
                SELECT actual_class, COUNT(*) 
                FROM predictions 
                WHERE actual_class IS NOT NULL
                GROUP BY actual_class
            """)
            class_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_predictions': total_predictions,
                'total_corrections': total_corrections,
                'correction_rate': total_corrections / total_predictions if total_predictions > 0 else 0,
                'accuracy': accuracy,
                'class_distribution': class_distribution
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {}
    
    def save_for_retraining(self, image_path: str, correct_class: str, correction_images_dir: str = 'correction_images') -> bool:
        """Save an image to the retraining folder with correct class organization"""
        try:
            import shutil
            from pathlib import Path
            
            # Create class directory if it doesn't exist
            class_dir = Path(correction_images_dir) / correct_class
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{os.path.basename(image_path)}"
            dest_path = class_dir / filename
            
            # Copy the image
            shutil.copy2(image_path, dest_path)
            
            logger.info(f"Saved image for retraining: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image for retraining: {e}")
            return False

# Convenience functions for easy import and use
def save_prediction(image_path: str, predicted_class: str, confidence: float) -> int:
    """Convenience function to save a prediction"""
    db = TriathlonDatabase()
    return db.save_prediction(image_path, predicted_class, confidence)

def save_correction(image_path: str, predicted_class: str, actual_class: str) -> bool:
    """Convenience function to save a correction"""
    db = TriathlonDatabase()
    return db.save_correction_by_image(image_path, predicted_class, actual_class)

def save_for_retraining(image_path: str, correct_class: str) -> bool:
    """Convenience function to save image for retraining"""
    db = TriathlonDatabase()
    return db.save_for_retraining(image_path, correct_class)

def get_corrections_count() -> int:
    """Convenience function to get corrections count"""
    db = TriathlonDatabase()
    return db.get_corrections_count()

def get_prediction_stats() -> dict:
    """Convenience function to get prediction statistics"""
    db = TriathlonDatabase()
    return db.get_prediction_stats()