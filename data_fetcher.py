import os
import time
import requests

# =========================
# CONFIGURATION
# =========================
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # REQUIRED
IMAGE_SIZE = "224x224"
ZOOM = 18
MAP_TYPE = "satellite"

BASE_IMAGE_DIR = "/content/drive/MyDrive/Satellite Imagery Based Property Valuation/data/images"


def fetch_image(lat, lon, save_path):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": IMAGE_SIZE,
        "maptype": MAP_TYPE,
        "key": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        pass

    return False


def fetch_images_from_dataframe(
    df,
    split="train",
    max_images=None,
    sleep_time=0.05
):
    assert split in ["train", "test"]

    save_dir = os.path.join(BASE_IMAGE_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    existing_ids = {
        f.replace(".png", "")
        for f in os.listdir(save_dir)
        if f.endswith(".png")
    }

    downloaded = 0
    skipped = 0

    for _, row in df.iterrows():
        img_id = str(row["id"])
        save_path = os.path.join(save_dir, f"{img_id}.png")

        if max_images and downloaded >= max_images:
            break

        if img_id in existing_ids:
            skipped += 1
            continue

        success = fetch_image(row["lat"], row["long"], save_path)

        if success:
            downloaded += 1
            if downloaded % 200 == 0:
                print(f"[{split}] Downloaded {downloaded} images")

        time.sleep(sleep_time)

    print(f"\n[{split.upper()} SUMMARY]")
    print(f"New images downloaded: {downloaded}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Total images now: {len(os.listdir(save_dir))}")
