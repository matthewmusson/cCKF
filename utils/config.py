import yaml
from pathlib import Path
import argparse
import hashlib
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_seed_string(seed_str: str) -> int:
    try:
        seed = int(seed_str)
        seed = abs(seed) % 900000000
        return 1 if seed == 0 else seed
    except ValueError:
        hash_obj = hashlib.md5(seed_str.encode())
        seed = int.from_bytes(hash_obj.digest()[:4], 'big', signed=True)
        seed = abs(seed) % 900000000
        return 1 if seed == 0 else seed


def create_base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", "-o", help="Output directory", type=Path, default=None)
    parser.add_argument("--events", "-n", help="Number of events", type=int, default=None)
    parser.add_argument("--config", help="YAML configuration file", type=Path, default=None)
    parser.add_argument("--seed", help="Random seed", type=str, default=None)
    parser.add_argument("--output-subdir", help="Output subdirectory", type=str, default=None)
    parser.add_argument("--performance-metrics", help="Enable performance metrics",
                        action="store_true", default=None)
    return parser


def load_config(args):
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        for key, value in config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    if args.seed is not None:
        args.seed = hash_seed_string(str(args.seed))
    else:
        args.seed = abs(int(time.time())) % 900000000
        args.seed = 1 if args.seed == 0 else args.seed

    return args
