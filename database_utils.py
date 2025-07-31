import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

class TriathlonDatabase:
    """Utility class for interacting with the triathlon predictions database"""
    
    def __init__(self, db_path: str = 'triathlon_predictions.db'):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def save_prediction(self, image_path: str, predicted_class: str, confidence: float) -> int:
        """Save a new prediction to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (image_path, predicted_class, confidence)
            VALUES (?, ?, ?)
        """, (image_path, predicted_class, confidence))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def save_correction(self, prediction_id: int, actual_class: str) -> bool:
        """Save a user correction for a prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions 
            SET actual_class = ?, corrected_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (actual_class, prediction_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_correction_count(self) -> int:
        """Get the number of corrections available for retraining"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_class IS NOT NULL")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_corrections_for_retraining(self) -> List[Dict[str, Any]]:
        """Get all corrections for model retraining"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_path, predicted_class, actual_class, confidence, corrected_at
            FROM predictions 
            WHERE actual_class IS NOT NULL
            ORDER BY corrected_at DESC
        """)
        
        corrections = []
        for row in cursor.fetchall():
            corrections.append({
                'image_path': row[0],
                'predicted_class': row[1],
                'actual_class': row[2],
                'confidence': row[3],
                'corrected_at': row[4]
            })
        
        conn.close()
        return corrections
    
    def add_model_version(self, version: int, model_path: str, accuracy: float) -> bool:
        """Add a new model version"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Deactivate current active model
            cursor.execute("UPDATE model_versions SET is_active = FALSE WHERE is_active = TRUE")
            
            # Add new model version
            cursor.execute("""
                INSERT INTO model_versions (version, model_path, accuracy, is_active)
                VALUES (?, ?, ?, TRUE)
            """, (version, model_path, accuracy))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get the currently active model version"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT version, model_path, accuracy, training_date
            FROM model_versions 
            WHERE is_active = TRUE
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'version': row[0],
                'model_path': row[1],
                'accuracy': row[2],
                'training_date': row[3]
            }
        return None
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics about predictions and corrections"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Total corrections
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_class IS NOT NULL")
        total_corrections = cursor.fetchone()[0]
        
        # Accuracy of current model (where corrections exist)
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN predicted_class = actual_class THEN 1 ELSE 0 END) as correct
            FROM predictions 
            WHERE actual_class IS NOT NULL
        """)
        accuracy_data = cursor.fetchone()
        
        current_accuracy = 0.0
        if accuracy_data[0] > 0:
            current_accuracy = accuracy_data[1] / accuracy_data[0]
        
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
            'current_accuracy': current_accuracy,
            'class_distribution': class_distribution
        }

# Example usage functions
def example_usage():
    """Example of how to use the database utilities"""
    db = TriathlonDatabase()
    
    # Save a prediction
    pred_id = db.save_prediction("images/test1.jpg", "bike", 0.85)
    print(f"Saved prediction with ID: {pred_id}")
    
    # Save a correction
    success = db.save_correction(pred_id, "run")
    print(f"Correction saved: {success}")
    
    # Get stats
    stats = db.get_prediction_stats()
    print(f"Database stats: {stats}")
    
    # Get active model
    active_model = db.get_active_model()
    print(f"Active model: {active_model}")

if __name__ == "__main__":
    example_usage()