# retrain_pipeline.py
from fastai.vision.all import *
import sqlite3
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TriathlonRetrainingPipeline:
    def __init__(self, db_path: str = 'triathlon_predictions.db', 
                 model_dir: str = 'models',
                 data_dir: str = 'training_data'):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.correction_images_dir = Path('correction_images')
        
        # Create directories if they don't exist
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.correction_images_dir.mkdir(exist_ok=True)
    
    def get_corrections_count(self) -> int:
        """Get count of user corrections since last retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the last retraining date
            cursor.execute("""
                SELECT MAX(training_date) FROM model_versions
            """)
            last_training = cursor.fetchone()[0]
            
            if last_training:
                # Count corrections since last training
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE actual_class IS NOT NULL 
                    AND corrected_at > ?
                """, (last_training,))
            else:
                # Count all corrections if no previous training
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE actual_class IS NOT NULL
                """)
            
            count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"Found {count} corrections available for retraining")
            return count
            
        except sqlite3.Error as e:
            logger.error(f"Database error getting corrections count: {e}")
            return 0
    
    def get_current_model_info(self) -> Tuple[int, float, str]:
        """Get current active model version, accuracy, and path"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT version, accuracy, model_path 
                FROM model_versions 
                WHERE is_active = TRUE
            """)
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0], result[1], result[2]
            else:
                logger.warning("No active model found in database")
                return 1, 0.895, 'models/triathlon_stage_model.pth'
                
        except sqlite3.Error as e:
            logger.error(f"Database error getting current model: {e}")
            return 1, 0.895, 'models/triathlon_stage_model.pth'
    
    def collect_correction_data(self) -> bool:
        """Collect corrected images and organize them for training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get corrections since last training
            current_version, _, _ = self.get_current_model_info()
            
            cursor.execute("""
                SELECT image_path, actual_class, corrected_at
                FROM predictions 
                WHERE actual_class IS NOT NULL
                ORDER BY corrected_at DESC
            """)
            corrections = cursor.fetchall()
            conn.close()
            
            if not corrections:
                logger.info("No corrections found")
                return False
            
            # Organize corrections by class
            classes = ['bike', 'run', 'swim', 'transition']
            
            # Create training structure
            train_dir = self.data_dir / 'train'
            valid_dir = self.data_dir / 'valid'
            
            for class_name in classes:
                (train_dir / class_name).mkdir(parents=True, exist_ok=True)
                (valid_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copy correction images to training folders
            correction_count = 0
            for image_path, actual_class, corrected_at in corrections:
                if os.path.exists(image_path):
                    # 80% to train, 20% to validation
                    dest_dir = train_dir if correction_count % 5 != 0 else valid_dir
                    dest_path = dest_dir / actual_class / f"correction_{correction_count}_{os.path.basename(image_path)}"
                    
                    shutil.copy2(image_path, dest_path)
                    correction_count += 1
                    logger.info(f"Copied {image_path} to {dest_path}")
            
            logger.info(f"Organized {correction_count} correction images for training")
            return correction_count > 0
            
        except Exception as e:
            logger.error(f"Error collecting correction data: {e}")
            return False
    
    def merge_with_original_dataset(self):
        """Merge correction data with original training data if available"""
        # This would merge with the original dataset if it exists
        # For now, we'll work with just the corrections
        logger.info("Using correction data for retraining")
    
    def train_model(self) -> Tuple[Optional[Learner], float]:
        """Train a new model with the collected data"""
        try:
            logger.info("Starting model training...")
            
            # Create data loaders
            dls = ImageDataLoaders.from_folder(
                self.data_dir,
                train='train',
                valid='valid',
                item_tfms=Resize(224),
                batch_tfms=aug_transforms(size=224, min_scale=0.75)
            )
            
            # Create learner
            learn = vision_learner(dls, resnet18, metrics=error_rate)
            
            # Find optimal learning rate
            learn.lr_find()
            
            # Fine-tune the model
            learn.fine_tune(5, freeze_epochs=2)
            
            # Get validation accuracy
            validation_accuracy = 1 - learn.validate()[1]  # 1 - error_rate = accuracy
            
            logger.info(f"Training completed. Validation accuracy: {validation_accuracy:.4f}")
            
            return learn, validation_accuracy
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return None, 0.0
    
    def evaluate_model(self, learn: Learner) -> float:
        """Evaluate the trained model"""
        try:
            # Run validation
            validation_results = learn.validate()
            accuracy = 1 - validation_results[1]  # 1 - error_rate
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def save_new_model(self, learn: Learner, version: int) -> str:
        """Save the new model version"""
        try:
            model_filename = f'triathlon_stage_model_v{version}.pth'
            model_path = self.model_dir / model_filename
            
            learn.export(model_path)
            
            logger.info(f"Model saved as {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def update_database(self, new_version: int, model_path: str, accuracy: float) -> bool:
        """Update database with new model version"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Deactivate current model
            cursor.execute("UPDATE model_versions SET is_active = FALSE")
            
            # Add new model version
            cursor.execute("""
                INSERT INTO model_versions (version, model_path, accuracy, is_active)
                VALUES (?, ?, ?, TRUE)
            """, (new_version, model_path, accuracy))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database updated with new model version {new_version}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error updating model version: {e}")
            return False
    
    def deploy_model(self, model_path: str) -> bool:
        """Deploy new model to production (copy to main model location)"""
        try:
            production_path = self.model_dir / 'triathlon_stage_model.pth'
            shutil.copy2(model_path, production_path)
            
            logger.info(f"Model deployed to production: {production_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False
    
    def cleanup_old_models(self, keep_versions: int = 3):
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old model versions to remove
            cursor.execute("""
                SELECT version, model_path FROM model_versions 
                WHERE is_active = FALSE 
                ORDER BY version DESC 
                LIMIT -1 OFFSET ?
            """, (keep_versions,))
            
            old_models = cursor.fetchall()
            
            for version, model_path in old_models:
                # Remove model file
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                # Remove from database
                cursor.execute("DELETE FROM model_versions WHERE version = ?", (version,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {len(old_models)} old model versions")
            
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
    
    def retrain_model(self, min_corrections: int = 50) -> str:
        """Main retraining pipeline function"""
        logger.info("=== Starting Automated Retraining Pipeline ===")
        
        # 1. Check if enough new data
        corrections_count = self.get_corrections_count()
        
        if corrections_count < min_corrections:
            message = f"Not enough data for retraining. Found {corrections_count} corrections, need {min_corrections}"
            logger.info(message)
            return message
        
        # 2. Get current model info
        current_version, current_accuracy, current_model_path = self.get_current_model_info()
        logger.info(f"Current model: v{current_version}, accuracy: {current_accuracy:.4f}")
        
        # 3. Collect and organize correction data
        if not self.collect_correction_data():
            message = "Failed to collect correction data"
            logger.error(message)
            return message
        
        # 4. Merge with original dataset
        self.merge_with_original_dataset()
        
        # 5. Train new model
        learn, new_accuracy = self.train_model()
        
        if learn is None:
            message = "Model training failed"
            logger.error(message)
            return message
        
        # 6. Evaluate performance
        final_accuracy = self.evaluate_model(learn)
        
        # 7. If better, deploy new model
        if final_accuracy > current_accuracy:
            new_version = current_version + 1
            model_path = self.save_new_model(learn, new_version)
            
            if model_path and self.update_database(new_version, model_path, final_accuracy):
                if self.deploy_model(model_path):
                    self.cleanup_old_models()
                    message = f"Retrained successfully! New model v{new_version} with accuracy: {final_accuracy:.4f} (improved from {current_accuracy:.4f})"
                    logger.info(message)
                    return message
                else:
                    message = "Model training successful but deployment failed"
                    logger.error(message)
                    return message
            else:
                message = "Model training successful but saving/database update failed"
                logger.error(message)
                return message
        else:
            message = f"New model accuracy ({final_accuracy:.4f}) not better than current ({current_accuracy:.4f}). Keeping current model."
            logger.info(message)
            return message

def run_retraining_pipeline():
    """Standalone function to run the retraining pipeline"""
    pipeline = TriathlonRetrainingPipeline()
    return pipeline.retrain_model()

if __name__ == "__main__":
    # Run the retraining pipeline
    result = run_retraining_pipeline()
    print(result)