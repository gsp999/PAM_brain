"""Index acquisition files and their log timestamps."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


DEFAULT_FILENAME_REGEX = r"(?P<index>\d+)_Col2_tdT_58_scan(?P<scan>\d+)\.dat"
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
                  log_files: list[Path] | None = None,
                  filename_regex: str = DEFAULT_FILENAME_REGEX) -> list[Record]:
    """Find .dat acquisitions; a custom regex needs an ``index`` group.

    The optional ``scan`` group separates acquisition batches for conservative
    adjacent recovery. Without it, all files belong to scan 1.
    """
    try:
        pattern = re.compile(filename_regex)
    except re.error as error:
        raise ValueError(f"Invalid filename regex: {error}") from error
    if "index" not in pattern.groupindex:
        raise ValueError("Filename regex must contain a named 'index' group")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    stamps: dict[int, datetime] = {}
    if log_dir is not None:
        for path in sorted(log_dir.glob("Col2_tdT_58_scan*_log.txt")):
            stamps.update(read_log(path))
    for path in log_files or []:
        stamps.update(read_log(path))
    records = []
    for path in data_dir.glob("*.dat"):
        match = pattern.fullmatch(path.name)
        if match:
            index = int(match.group("index"))
            scan_text = match.group("scan") if "scan" in pattern.groupindex else None
            scan = int(scan_text) if scan_text is not None else 1
            records.append(Record(index, scan, path.resolve(), stamps.get(index)))
    records.sort(key=lambda r: r.index)
    if len({r.index for r in records}) != len(records):
        raise ValueError("Duplicate acquisition indices")
    return records
