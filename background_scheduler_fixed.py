"""
Background scheduler for automated daily data processing and model training - FIXED VERSION
"""

import schedule
import time
import threading
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional
import psutil
import os
import json

from incremental_training_system import incremental_system
from utils.logger import get_logger
from config import get_config

logger = get_logger(__name__)


class BackgroundScheduler:
    """Background scheduler for automated data processing"""

    def __init__(self, daily_time: str = "02:00"):
        """
        Initialize scheduler

        Args:
            daily_time: Time to run daily processing (HH:MM format)
        """
        self.daily_time = daily_time
        self.is_running = False
        self.scheduler_thread = None
        self.last_run_file = Path("cache/last_run.txt")
        self.lock_file = Path("cache/scheduler.lock")

        # Ensure cache directory exists
        Path("cache").mkdir(exist_ok=True)

        # Setup signal handlers safely
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers safely - only in main thread"""
        try:
            # Only setup signal handlers if we're in the main thread
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                logger.info("Signal handlers setup successfully")
            else:
                logger.info("Not in main thread - skipping signal handler setup")
        except ValueError as e:
            logger.warning(f"Could not setup signal handlers: {e}")
            # This is fine - signal handlers are optional

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def _create_lock(self) -> bool:
        """Create lock file to prevent multiple instances"""
        if self.lock_file.exists():
            try:
                # Check if the process in the lock file is still running
                with open(self.lock_file, "r") as f:
                    pid = int(f.read().strip())

                if psutil.pid_exists(pid):
                    logger.warning(f"Another scheduler instance is running (PID: {pid})")
                    return False
                else:
                    # Process is dead, remove stale lock
                    self.lock_file.unlink()
                    logger.info("Removed stale lock file")
            except (ValueError, FileNotFoundError):
                # Invalid or missing lock file, remove it
                self.lock_file.unlink()

        # Create new lock file
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        return True

    def _remove_lock(self):
        """Remove lock file"""
        if self.lock_file.exists():
            self.lock_file.unlink()

    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        if not self._create_lock():
            return

        logger.info(
            f"🚀 Starting background scheduler with daily processing at {self.daily_time}"
        )

        # Schedule daily processing
        schedule.every().day.at(self.daily_time).do(self.daily_processing_job)

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("✅ Background scheduler started successfully")

    def stop(self):
        """Stop the background scheduler"""
        if not self.is_running:
            return

        logger.info("🛑 Stopping background scheduler...")
        self.is_running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        self._remove_lock()
        logger.info("✅ Background scheduler stopped")

    def _run_scheduler(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)

    def daily_processing_job(self):
        """Main daily processing job"""
        start_time = datetime.now()
        logger.info(f"🚀 Starting daily processing job at {start_time}")

        try:
            # Update last run timestamp
            with open(self.last_run_file, "w") as f:
                f.write(start_time.isoformat())

            # Run incremental update
            logger.info("Running incremental data update...")
            features_data, forecaster = incremental_system.incremental_update()

            # Log results
            num_products = (
                features_data["product_id"].nunique() if not features_data.empty else 0
            )
            num_records = len(features_data)

            processing_time = datetime.now() - start_time

            logger.info(f"✅ Daily processing completed successfully!")
            logger.info(
                f"📊 Processed {num_records:,} records for {num_products:,} products"
            )
            logger.info(f"⏱️ Processing time: {processing_time}")

            # Create status report
            self._create_status_report(
                start_time, processing_time, num_records, num_products, True
            )

        except Exception as e:
            error_msg = f"❌ Daily processing failed: {str(e)}"
            logger.error(error_msg)

            processing_time = datetime.now() - start_time
            self._create_status_report(
                start_time, processing_time, 0, 0, False, str(e)
            )

            # Don't raise exception - continue running scheduler

    def _create_status_report(
        self,
        start_time: datetime,
        processing_time: timedelta,
        num_records: int,
        num_products: int,
        success: bool,
        error_msg: Optional[str] = None,
    ):
        """Create processing status report"""
        report = {
            "timestamp": start_time.isoformat(),
            "processing_time_seconds": processing_time.total_seconds(),
            "num_records": num_records,
            "num_products": num_products,
            "success": success,
            "error_message": error_msg,
        }

        # Save report
        report_file = Path("cache/last_processing_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Also create a simple status file for the web app
        status_file = Path("cache/processing_status.txt")
        status = "SUCCESS" if success else "FAILED"
        with open(status_file, "w") as f:
            f.write(f"{status}\n{start_time.isoformat()}\n{num_records}\n{num_products}")

    def force_run_now(self):
        """Force run the daily processing job immediately"""
        logger.info("🚀 Forcing immediate processing run...")
        threading.Thread(target=self.daily_processing_job, daemon=True).start()

    def get_status(self) -> dict:
        """Get current scheduler status"""
        return {
            "is_running": self.is_running,
            "daily_time": self.daily_time,
            "last_run": self._get_last_run(),
            "next_run": self._get_next_run(),
        }

    def _get_last_run(self) -> Optional[str]:
        """Get last run timestamp"""
        if self.last_run_file.exists():
            try:
                with open(self.last_run_file, "r") as f:
                    return f.read().strip()
            except Exception:
                return None
        return None

    def _get_next_run(self) -> Optional[str]:
        """Get next scheduled run time"""
        if schedule.jobs:
            next_run = schedule.next_run()
            if next_run:
                return next_run.isoformat()
        return None

    def get_last_processing_report(self) -> Optional[dict]:
        """Get the last processing report"""
        report_file = Path("cache/last_processing_report.json")
        if report_file.exists():
            try:
                with open(report_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading processing report: {e}")
        return None


# Function to get scheduler instance safely
def get_background_scheduler(daily_time: str = "02:00") -> BackgroundScheduler:
    """Get a new background scheduler instance safely"""
    return BackgroundScheduler(daily_time)


# Global scheduler instance - but only create when needed
_background_scheduler = None


def get_global_scheduler(daily_time: str = "02:00") -> BackgroundScheduler:
    """Get or create the global background scheduler instance"""
    global _background_scheduler
    if _background_scheduler is None:
        _background_scheduler = BackgroundScheduler(daily_time)
    return _background_scheduler


# For backward compatibility - but don't instantiate immediately
background_scheduler = None


def init_background_scheduler(daily_time: str = "02:00"):
    """Initialize the global background scheduler"""
    global background_scheduler
    if background_scheduler is None:
        background_scheduler = BackgroundScheduler(daily_time)
    return background_scheduler
