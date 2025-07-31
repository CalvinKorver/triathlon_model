import sqlite3
import os
from datetime import datetime

def create_database():
    """Create SQLite database with required tables for Phase 3 triathlon model feedback loop"""
    
    db_path = 'triathlon_predictions.db'
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create new database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_class TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            corrected_at TIMESTAMP
        )
    """)
    
    # Create model_versions table
    cursor.execute("""
        CREATE TABLE model_versions (
            version INTEGER PRIMARY KEY,
            model_path TEXT NOT NULL,
            accuracy REAL NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX idx_predictions_created_at ON predictions(created_at)")
    cursor.execute("CREATE INDEX idx_predictions_corrected ON predictions(corrected_at)")
    cursor.execute("CREATE INDEX idx_model_versions_active ON model_versions(is_active)")
    
    # Insert initial model version
    cursor.execute("""
        INSERT INTO model_versions (version, model_path, accuracy, is_active)
        VALUES (1, 'models/triathlon_stage_model.pth', 0.895, TRUE)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Database created successfully: {db_path}")
    print("Tables created: predictions, model_versions")
    print("Initial model version (v1) added with 89.5% accuracy")

def get_database_info():
    """Display information about the database structure"""
    
    db_path = 'triathlon_predictions.db'
    
    if not os.path.exists(db_path):
        print("Database does not exist. Run create_database() first.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"Database: {db_path}")
    print(f"Tables: {[table[0] for table in tables]}")
    
    for table_name in [table[0] for table in tables]:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"\n{table_name} table structure:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    # Show current model version
    cursor.execute("SELECT * FROM model_versions WHERE is_active = TRUE")
    active_model = cursor.fetchone()
    if active_model:
        print(f"\nActive model: Version {active_model[0]}, Accuracy: {active_model[2]:.3f}")
    
    conn.close()

if __name__ == "__main__":
    create_database()
    get_database_info()