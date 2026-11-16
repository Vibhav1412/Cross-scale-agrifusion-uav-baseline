"""
PhenoBench Dataset Resumable Downloader
Source: Official University of Bonn PhenoBench repository (https://www.phenobench.org/data/PhenoBench-v110.zip)
Size: ~7.63 GB (7,630,658,167 bytes)
License: CC-BY-SA 4.0
"""

import os
import sys
import time
import argparse
import urllib.request

DEFAULT_URL = "https://www.phenobench.org/data/PhenoBench-v110.zip"
DEFAULT_DEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "phenobench", "PhenoBench-v110.zip"
)

def format_size(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

def format_time(seconds):
    if seconds is None or seconds < 0 or seconds > 86400 * 30:
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def download_file(url=DEFAULT_URL, dest_path=DEFAULT_DEST, chunk_size=1024 * 1024, max_retries=30):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    print(f"=" * 70)
    print("PHENOBENCH DATASET DOWNLOADER (RESUMABLE & AUTO-RETRYING)")
    print(f"URL:         {url}")
    print(f"Destination: {dest_path}")
    print(f"=" * 70)
    
    # 1. Query remote file size via HEAD
    head_req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(head_req, timeout=15) as head_resp:
            total_size = int(head_resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"[ERROR] Could not fetch file header: {e}")
        return False

    print(f"Total Remote File Size: {format_size(total_size)} ({total_size:,} bytes)")

    # 2. Outer loop for automatic retry / resumption
    retries = 0
    start_time = time.time()
    last_print = 0
    bytes_since_start = 0

    while retries < max_retries:
        downloaded = 0
        if os.path.exists(dest_path):
            downloaded = os.path.getsize(dest_path)
            if downloaded >= total_size and total_size > 0:
                print(f"\n[OK] File already fully downloaded: {dest_path} ({format_size(downloaded)})")
                return True
            elif downloaded > 0:
                print(f"\n[RESUME] Connecting with Range: bytes={downloaded:,}-{total_size-1:,} ({format_size(downloaded)})...")

        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgriFusionDownloader/1.0")
        if downloaded > 0:
            req.add_header("Range", f"bytes={downloaded}-{total_size - 1}")

        open_mode = "ab" if downloaded > 0 else "wb"

        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, open_mode) as out_file:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        # Remote stream closed; break inner loop to check if complete
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    bytes_since_start += len(chunk)

                    now = time.time()
                    if now - last_print >= 2.0 or downloaded >= total_size:
                        elapsed = now - start_time
                        speed = bytes_since_start / elapsed if elapsed > 0 else 0
                        pct = (downloaded / total_size * 100.0) if total_size > 0 else 0
                        remaining_bytes = max(0, total_size - downloaded)
                        eta = remaining_bytes / speed if speed > 0 else 0

                        print(
                            f"\rProgress: {pct:6.2f}% | "
                            f"{format_size(downloaded)} / {format_size(total_size)} | "
                            f"Speed: {format_size(speed)}/s | "
                            f"ETA: {format_time(eta)}",
                            end="", flush=True
                        )
                        last_print = now

            if downloaded >= total_size and total_size > 0:
                print("\n" + "=" * 70)
                print(f"[SUCCESS] Download completed successfully!")
                print(f"Saved at: {dest_path}")
                print(f"Final Size: {format_size(os.path.getsize(dest_path))}")
                print("=" * 70)
                return True
            else:
                retries += 1
                print(f"\n[INFO] Connection dropped by server at {format_size(downloaded)}. Auto-resuming (attempt {retries}/{max_retries}) in 3s...")
                time.sleep(3)

        except KeyboardInterrupt:
            print("\n[PAUSED] Download paused by user. Run the script again to resume.")
            return False
        except Exception as e:
            retries += 1
            print(f"\n[WARN] Connection error: {e}. Auto-resuming (attempt {retries}/{max_retries}) in 5s...")
            time.sleep(5)

    print(f"\n[FAIL] Reached maximum retries ({max_retries}). Run script again to resume.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhenoBench Dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="Download URL")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination zip path")
    args = parser.parse_args()
    
    download_file(url=args.url, dest_path=args.dest)

