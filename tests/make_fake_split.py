"""Write synthetic per-event Parquets for CLI smoke tests.

Usage: python tests/make_fake_split.py <out_dir> <event_id> [<event_id> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cckf.splits import SCHEMA_76
from tests.conftest import _rows


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(_rows())[list(SCHEMA_76)]
    for event_id in (int(a) for a in sys.argv[2:]):
        df = base.copy()
        df["event_id"] = event_id
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            out_dir / f"expanded_event{event_id:09d}.parquet",
        )
    print(f"wrote {len(sys.argv) - 2} files to {out_dir}")


if __name__ == "__main__":
    main()
