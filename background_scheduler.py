"""
Background scheduler for automated daily data processing and model training
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

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def _create_lock(self) -> bool:
        """Create lock file to prevent multiple instances"""
        if self.lock_file.exists():
            # Check if process is still running
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())

                if psutil.pid_exists(old_pid):
                    logger.warning(f"Scheduler already running with PID {old_pid}")
                    return False
                else:
                    logger.info("Removing stale lock file")
                    self.lock_file.unlink()
            except:
                self.lock_file.unlink()

        # Create new lock file
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        return True

    def _remove_lock(self):
        """Remove lock file"""
        if self.lock_file.exists():
            self.lock_file.unlink()

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
            self._create_status_report(start_time, processing_time, 0, 0, False, str(e))

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
        """Create status report file"""
        status_file = Path("cache/last_processing_status.json")

        status = {
            "timestamp": start_time.isoformat(),
            "processing_time_seconds": processing_time.total_seconds(),
            "success": success,
            "num_records": num_records,
            "num_products": num_products,
            "error_message": error_msg,
        }

        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

    def get_last_run_status(self) -> dict:
        """Get status of last processing run"""
        status_file = Path("cache/last_processing_status.json")

        if not status_file.exists():
            return {"status": "never_run"}

        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            return {"status": "error_reading_status"}

    def should_run_initial_processing(self) -> bool:
        """Check if we should run initial processing on startup"""
        if not self.last_run_file.exists():
            return True

        try:
            with open(self.last_run_file, "r") as f:
                last_run = datetime.fromisoformat(f.read().strip())

            # Run if last run was more than 24 hours ago
            return datetime.now() - last_run > timedelta(hours=24)

        except:
            return True

    def run_initial_processing_if_needed(self):
        """Run initial processing if needed"""
        if self.should_run_initial_processing():
            logger.info("Running initial processing on startup...")
            self.daily_processing_job()
        else:
            logger.info("Skipping initial processing - recent run found")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info(f"📅 Scheduler started - daily processing at {self.daily_time}")

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

        logger.info("Scheduler loop ended")

    def start(self, run_initial: bool = True):
        """Start the background scheduler"""
        if not self._create_lock():
            raise RuntimeError("Scheduler is already running")

        try:
            self.is_running = True

            # Schedule daily job
            schedule.every().day.at(self.daily_time).do(self.daily_processing_job)

            # Run initial processing if needed
            if run_initial:
                self.run_initial_processing_if_needed()

            # Start scheduler thread
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True
            )
            self.scheduler_thread.start()

            logger.info(f"✅ Background scheduler started successfully")
            logger.info(f"📅 Next run scheduled for: {schedule.next_run()}")

        except Exception as e:
            self._remove_lock()
            raise e

    def stop(self):
        """Stop the background scheduler"""
        logger.info("Stopping background scheduler...")

        self.is_running = False

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

        schedule.clear()
        self._remove_lock()

        logger.info("✅ Background scheduler stopped")

    def get_next_run_time(self) -> Optional[datetime]:
        """Get next scheduled run time"""
        next_run = schedule.next_run()
        return next_run if next_run else None

    def force_run_now(self):
        """Force run processing job now (for testing/manual trigger)"""
        logger.info("🔥 Forcing immediate processing run...")
        self.daily_processing_job()


# Global scheduler instance
background_scheduler = BackgroundScheduler()
