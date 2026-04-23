import requests
import os

url = "https://data.insideairbnb.com/united-states/ny/new-york-city/2024-06-05/data/listings.csv.gz"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print("Downloading NYC Airbnb listings data...")
response = requests.get(url, headers=headers, timeout=120)
response.raise_for_status()

os.makedirs("data", exist_ok=True)
filepath = "data/listings.csv.gz"

with open(filepath, "wb") as f:
    f.write(response.content)

size_mb = round(os.path.getsize(filepath) / 1e6, 1)
print(f"Done! Saved to {filepath} ({size_mb} MB)")
