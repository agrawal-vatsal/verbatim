import json
import httpx
import time
from pathlib import Path


def download_transcripts() -> None:
    transcript_dir = Path("data/raw/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Standard browser headers to bypass 403/406 errors
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }

    with open("manifest.json", "r") as f:
        data = json.load(f)

    transcripts = data.get("transcripts", [])

    # Increase max_redirects for L&T and add a timeout
    with httpx.Client(
            follow_redirects=True, headers=HEADERS, max_redirects=20, timeout=30.0
            ) as client:
        for entry in transcripts:
            filename = f"{entry['company']}_{entry['fy']}_{entry['quarter']}.pdf"
            filepath = transcript_dir / filename

            if filepath.exists():
                print(f"Skipping {filename}, already exists.")
                continue

            print(f"Downloading {filename}...")
            try:
                response = client.get(entry["url"])
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved to {filepath}")

                # Polite delay to avoid rate limiting
                time.sleep(1)

            except httpx.HTTPStatusError as e:
                print(
                    f"❌ Failed to download {filename}: {e.response.status_code} at {entry['url']}"
                    )
            except httpx.TooManyRedirects:
                print(
                    f"❌ Failed {filename}: Too many redirects. Check URL or session requirements."
                    )
            except Exception as e:
                print(f"❌ Unexpected error for {filename}: {str(e)}")


if __name__ == "__main__":
    download_transcripts()
