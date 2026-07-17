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
        self.acts_timings = []
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
            duration = time.time() - start
            self.timings[name] = duration
            self.logger.info(f"Completed stage: {name} in {duration:.2f} seconds")

    def parse_acts_timing(self, tsv_path):
        """Parse ACTS Sequencer timing.tsv into per-algorithm breakdown."""
        try:
            with open(tsv_path) as f:
                header = f.readline().strip().split("\t")
                for line in f:
                    fields = line.strip().split("\t")
                    if len(fields) != len(header):
                        continue
                    row = dict(zip(header, fields))
                    self.acts_timings.append(row)
        except Exception as e:
            self.logger.warning(f"Could not parse {tsv_path}: {e}")

    def write_report(self):
        try:
            total_time = time.time() - self.start_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            w = 50

            report = [f"Timing Report ({timestamp})", "=" * w]
            if self.error_occurred:
                report.append("*** Errors occurred during execution ***")

            report.append("Pipeline stages:")
            for name, duration in self.timings.items():
                report.append(f"  {name:<40} {duration:>7.2f}s")
            report.append(f"  {'Total':<40} {total_time:>7.2f}s")

            if self.acts_timings:
                # Identify key columns
                name_col = "identifier" if "identifier" in self.acts_timings[0] else "name"
                time_col = None
                for candidate in ["time_perevent_s", "time_total_s", "total_time_s"]:
                    if candidate in self.acts_timings[0]:
                        time_col = candidate
                        break

                if time_col:
                    report.append("")
                    per_event = "perevent" in time_col
                    label = "per-event" if per_event else "total"
                    report.append(f"ACTS algorithm breakdown ({label}):")
                    report.append("-" * w)
                    for row in self.acts_timings:
                        name = row.get(name_col, "?")
                        t = float(row.get(time_col, 0))
                        report.append(f"  {name:<40} {t:>7.4f}s")

            if self.errors:
                report.append(f"\n{'Errors':}")
                for error in self.errors:
                    report.append(f"  {error}")

            report_text = "\n".join(report)
            summary_path = self.output_dir / "timing_summary.txt"
            with open(summary_path, "w") as f:
                f.write(report_text)
            print(report_text)
        except Exception as e:
            print(f"Error writing timing report: {str(e)}")
            print(traceback.format_exc())
