import os
import requests
import zipfile
import io
import time

# Configuration
DATA_DIR = "data"

# Shot-level data, hosted on MoneyPuck's download backend.
# Each zip extracts to shots_<year>.csv (~70 MB), which feature_engineering.py reads.
BASE_URL_SHOTS = "https://peter-tanner.com/moneypuck/downloads"

# We fetch two seasons: 2024 and 2025.
DOWNLOAD_TASKS = [
    {
        "url": f"{BASE_URL_SHOTS}/shots_2024.zip",
        "filename": "shots_2024.zip",
        "type": "zip",
        "desc": "Shot Data 2024"
    },
    {
        "url": f"{BASE_URL_SHOTS}/shots_2025.zip",
        "filename": "shots_2025.zip",
        "type": "zip",
        "desc": "Shot Data 2025"
    },
]

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_file(task):
    url = task['url']
    filename = task['filename']
    local_path = os.path.join(DATA_DIR, filename)

    # Check if file already exists to avoid re-downloading large zips
    if os.path.exists(local_path):
        print(f"[Skipping] {task['desc']} (File already exists)")
        return

    print(f"\n[Downloading] {task['desc']}...")
    print(f"   Source: {url}")

    try:
        # Add headers to mimic a browser request to avoid some 403/404s
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        if task['type'] == 'zip':
            print("   Extracting archive...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(DATA_DIR)
            print(f"   ✓ Extracted to {DATA_DIR}/")
            # We keep the zip file so we don't download it again next time
            with open(local_path, 'wb') as f:
                f.write(response.content)

        else:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✓ Saved to {local_path}")

        # Be nice to the server
        time.sleep(1)

    except requests.exceptions.HTTPError as e:
        print(f"   X Failed to download {filename}: {e}")
    except Exception as e:
        print(f"   X Error: {e}")

def main():
    print("--- NHL Goalie Model Data Fetcher (v2) ---")
    ensure_directory(DATA_DIR)

    for task in DOWNLOAD_TASKS:
        download_file(task)

    print("\n--- Download Process Complete ---")
    print(f"Check the '{DATA_DIR}' folder for your files.")

if __name__ == "__main__":
    main()