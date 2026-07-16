import logging
from pathlib import Path
import time
import traceback
from contextlib import contextmanager


def setup_logging(name="CKF_Chain", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)-12s %(message)s'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class TimingRecorder:
    def __init__(self, output_dir):
        self.timings = {}
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.errors = []
        self.error_occurred = False
        self.logger = logging.getLogger("TimingRecorder")

    @contextmanager
    def record(self, name):
        self.logger.info(f"Starting stage: {name}")
        start = time.time()
        try:
            yield
        except Exception as e:
            self.errors.append(f"Error in {name}: {str(e)}")
            self.error_occurred = True
            raise
        finally:
            end = time.time()
            duration = end - start
            self.timings[name] = duration
            self.logger.info(f"Completed stage: {name} in {duration:.2f} seconds")

    def write_report(self):
        try:
            total_time = time.time() - self.start_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report = [f"Timing Report ({timestamp})", "============="]
            if self.error_occurred:
                report.append("*** Errors occurred during execution ***")
            for name, duration in sorted(self.timings.items()):
                report.append(f"{name:<30} : {duration:>.2f} seconds")
            report.append("-" * 50)
            report.append(f"{'Total time':<30} : {total_time:>.2f} seconds")
            if self.errors:
                report.append("\nErrors encountered:")
                report.append("===================")
                for error in self.errors:
                    report.append(error)
            summary_path = self.output_dir / "timing_summary.txt"
            with open(summary_path, "a") as f:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("\n".join(report))
            print("\n".join(report))
        except Exception as e:
            print(f"Error writing timing report: {str(e)}")
            print(traceback.format_exc())
