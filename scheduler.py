#!/usr/bin/env python3
"""
Scheduler for automated triathlon model retraining
Supports both cron-style scheduling and manual execution
"""

import schedule
import time
import logging
from datetime import datetime
import argparse
import sys
from retrain_pipeline import run_retraining_pipeline
from database_helpers import get_corrections_count, get_prediction_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_scheduled_retraining():
    """Run the retraining pipeline with logging and error handling"""
    logger.info("=== Scheduled Retraining Job Started ===")
    
    try:
        # Log current stats before retraining
        stats = get_prediction_stats()
        logger.info(f"Current stats: {stats}")
        
        # Run retraining pipeline
        result = run_retraining_pipeline()
        logger.info(f"Retraining result: {result}")
        
        # Log stats after retraining
        new_stats = get_prediction_stats()
        logger.info(f"Updated stats: {new_stats}")
        
    except Exception as e:
        logger.error(f"Error during scheduled retraining: {e}")
    
    logger.info("=== Scheduled Retraining Job Completed ===")

def check_retraining_eligibility():
    """Check if retraining is needed and log status"""
    corrections_count = get_corrections_count()
    logger.info(f"Current corrections available: {corrections_count}")
    
    if corrections_count >= 50:
        logger.info("Sufficient corrections available for retraining")
        return True
    else:
        logger.info(f"Need {50 - corrections_count} more corrections for retraining")
        return False

def start_scheduler():
    """Start the weekly scheduler"""
    logger.info("Starting automated retraining scheduler...")
    
    # Schedule retraining every Sunday at 2 AM
    schedule.every().sunday.at("02:00").do(run_scheduled_retraining)
    
    # Schedule daily eligibility checks at 6 AM
    schedule.every().day.at("06:00").do(check_retraining_eligibility)
    
    logger.info("Scheduler configured:")
    logger.info("- Retraining: Every Sunday at 2:00 AM")
    logger.info("- Status check: Every day at 6:00 AM")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

def create_cron_job():
    """Generate cron job command for system-level scheduling"""
    import os
    
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    
    # Weekly retraining (Sundays at 2 AM)
    cron_command = f"0 2 * * 0 {python_path} {script_path} --run-retraining"
    
    print("To set up automatic retraining using system cron, add this line to your crontab:")
    print("(Run 'crontab -e' to edit)")
    print()
    print(cron_command)
    print()
    print("For daily status checks, also add:")
    print(f"0 6 * * * {python_path} {script_path} --check-status")

def main():
    parser = argparse.ArgumentParser(description="Triathlon Model Retraining Scheduler")
    
    parser.add_argument("--run-retraining", action="store_true",
                       help="Run retraining pipeline once")
    parser.add_argument("--check-status", action="store_true",
                       help="Check retraining eligibility")
    parser.add_argument("--start-scheduler", action="store_true",
                       help="Start continuous scheduler")
    parser.add_argument("--create-cron", action="store_true",
                       help="Show cron job setup instructions")
    parser.add_argument("--min-corrections", type=int, default=50,
                       help="Minimum corrections needed for retraining (default: 50)")
    
    args = parser.parse_args()
    
    if args.run_retraining:
        run_scheduled_retraining()
    elif args.check_status:
        check_retraining_eligibility()
    elif args.start_scheduler:
        start_scheduler()
    elif args.create_cron:
        create_cron_job()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()