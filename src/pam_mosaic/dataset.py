"""Index acquisition files and their log timestamps."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


NAME = re.compile(r"^(\d+)_Col2_tdT_58_scan(\d+)\.dat$")
ENTRY = re.compile(r"^(\d+)_(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}:\d{2})$")


@dataclass(frozen=True)
class Record:
    index: int
    scan: int
    path: Path
    timestamp: datetime | None = None


def read_log(path: Path) -> dict[int, datetime]:
    result: dict[int, datetime] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = ENTRY.fullmatch(line.strip())
        if match:
            result[int(match.group(1))] = datetime.strptime(match.group(2), "%Y/%m/%d %H:%M:%S")
    return result


def index_dataset(data_dir: Path, log_dir: Path | None = None,
                  log_files: list[Path] | None = None) -> list[Record]:
    stamps: dict[int, datetime] = {}
    if log_dir is not None:
        for path in sorted(log_dir.glob("Col2_tdT_58_scan*_log.txt")):
            stamps.update(read_log(path))
    for path in log_files or []:
        stamps.update(read_log(path))
    records = []
    for path in data_dir.glob("*.dat"):
        match = NAME.fullmatch(path.name)
        if match:
            index, scan = map(int, match.groups())
            records.append(Record(index, scan, path, stamps.get(index)))
    records.sort(key=lambda r: r.index)
    if len({r.index for r in records}) != len(records):
        raise ValueError("Duplicate acquisition indices")
    return records
