from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm.auto import tqdm

LICHESS_BASE_URL = "https://database.lichess.org/standard"
CHECKSUM_INDEX_URL = f"{LICHESS_BASE_URL}/sha256sums.txt"
CHUNK_SIZE = 8 * 1024 * 1024
REQUEST_TIMEOUT = (30, 300)
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    month: str
    path: Path
    bytes_downloaded: int
    checksum_ok: bool | None


def filename_for_month(month: str) -> str:
    if not MONTH_PATTERN.match(month):
        raise ValueError(f"Invalid month {month!r}, expected YYYY-MM (e.g. 2024-06)")
    return f"lichess_db_standard_rated_{month}.pgn.zst"


def url_for_month(month: str) -> str:
    return f"{LICHESS_BASE_URL}/{filename_for_month(month)}"


def parse_checksum_index(text: str) -> dict[str, str]:
    checksums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or line.startswith("#"):
            continue
        sha256, fname = parts[0].lower(), parts[-1]
        if len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256):
            checksums[fname] = sha256
    return checksums


def fetch_official_checksums() -> dict[str, str]:
    resp = requests.get(CHECKSUM_INDEX_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    checksums = parse_checksum_index(resp.text)
    logger.info("Loaded %d checksums from %s", len(checksums), CHECKSUM_INDEX_URL)
    return checksums


def compute_sha256(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                                    desc=f"SHA256 {path.name}", leave=False) as bar:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def stream_download(url: str, out_path: Path, resume_from: int = 0) -> int:
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"

    with requests.get(url, stream=True, headers=headers, timeout=REQUEST_TIMEOUT) as resp:
        if resume_from and resp.status_code == 200:
            logger.warning("Server ignored the Range header, starting from scratch")
            mode, resume_from = "wb", 0
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0)) + resume_from
        written = 0
        with out_path.open(mode) as f, tqdm(total=total or None, initial=resume_from, unit="B",
                                            unit_scale=True, desc=out_path.name) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
    return written


def download_month(month: str, output_dir: Path, checksums: dict[str, str] | None = None,
                   verify: bool = True) -> DownloadResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = filename_for_month(month)
    final_path = output_dir / fname
    part_path = output_dir / (fname + ".part")

    expected = None
    if verify:
        if checksums is None:
            checksums = fetch_official_checksums()
        expected = checksums.get(fname)
        if expected is None:
            logger.warning("No official checksum for %s, downloading without verification", fname)

    if final_path.exists():
        if expected is None:
            logger.info("%s already exists, skipping", fname)
            return DownloadResult(month, final_path, 0, None)
        if compute_sha256(final_path) == expected:
            logger.info("%s already exists and the checksum matches, skipping", fname)
            return DownloadResult(month, final_path, 0, True)
        logger.warning("%s exists but the checksum differs, downloading again", fname)
        final_path.unlink()

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if resume_from:
        logger.info("Resuming %s from byte %d", fname, resume_from)
    written = stream_download(url_for_month(month), part_path, resume_from)
    part_path.replace(final_path)
    logger.info("Downloaded %s (%d bytes)", final_path, written)

    checksum_ok = None
    if expected is not None:
        actual = compute_sha256(final_path)
        checksum_ok = actual == expected
        if checksum_ok:
            logger.info("SHA256 OK")
        else:
            logger.error("SHA256 mismatch: got %s, expected %s", actual, expected)
    return DownloadResult(month, final_path, written, checksum_ok)


def verify_month(month: str, output_dir: Path, checksums: dict[str, str]) -> bool:
    fname = filename_for_month(month)
    path = Path(output_dir) / fname
    if not path.exists():
        logger.error("Missing file: %s", path)
        return False
    expected = checksums.get(fname)
    if expected is None:
        logger.warning("No official checksum for %s", fname)
        return False
    ok = compute_sha256(path) == expected
    logger.info("%s: %s", fname, "OK" if ok else "MISMATCH")
    return ok


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download monthly game dumps from the Lichess database.")
    p.add_argument("--months", nargs="+", required=True, help="months as YYYY-MM, e.g. 2024-06")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="on Colab use local disk (/content/lichess_raw), Drive is too small")
    p.add_argument("--verify-only", action="store_true", help="only check SHA256 of existing files")
    p.add_argument("--no-verify", action="store_true", help="skip the SHA256 check")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.verify_only:
        checksums = fetch_official_checksums()
        results = [verify_month(m, args.output_dir, checksums) for m in args.months]
        return 0 if all(results) else 1

    checksums = None if args.no_verify else fetch_official_checksums()
    failed = []
    for month in args.months:
        result = download_month(month, args.output_dir, checksums, verify=not args.no_verify)
        if result.checksum_ok is False:
            failed.append(month)
    if failed:
        logger.error("Checksum failed for: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
