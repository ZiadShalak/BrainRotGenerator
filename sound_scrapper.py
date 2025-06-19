#!/usr/bin/env python3
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CATEGORY_URL = "https://www.myinstants.com/en/categories/sound%20effects/"
OUTPUT_DIR   = "sounds"
SCROLL_PAUSE = 2.5       # seconds to wait after each scroll
MAX_SCROLLS  = 100       # safety cap
MAX_WORKERS  = 10        # parallel downloads

# ─── SETUP ───────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

opts = Options()
opts.add_argument("--headless")
opts.add_argument("--disable-gpu")
opts.add_argument("--window-size=1920,1080")
# no anti-automation flags so JS can load all content
driver = webdriver.Chrome(options=opts)

# ─── SCROLL TO LOAD EVERYTHING ──────────────────────────────────────────────────
driver.get(CATEGORY_URL)
time.sleep(5)  # initial load

last_height = driver.execute_script("return document.body.scrollHeight")
for i in range(MAX_SCROLLS):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# ─── EXTRACT MP3 URLS ───────────────────────────────────────────────────────────
html = driver.page_source
driver.quit()

# find all “/media/sounds/…mp3” occurences
paths = re.findall(r'(/media/sounds/[A-Za-z0-9_\-\.]+\.mp3)', html)
mp3_urls = {
    "https://www.myinstants.com" + p
    for p in paths
}
print(f"🔍 Found {len(mp3_urls)} unique MP3 URLs.")

# ─── DOWNLOAD IN PARALLEL ───────────────────────────────────────────────────────
def download(item):
    idx, url = item
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        name = os.path.basename(url)
        out  = f"{idx:03}_{name}"
        with open(os.path.join(OUTPUT_DIR, out), "wb") as f:
            f.write(r.content)
        return idx, out, None
    except Exception as e:
        return idx, url, e

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    for idx, info, err in pool.map(download, enumerate(sorted(mp3_urls), 1)):
        if err is None:
            print(f"[{idx}/{len(mp3_urls)}] ✅ {info}")
        else:
            print(f"[{idx}/{len(mp3_urls)}] ❌ {info}: {err}")

print(f"\n🎉 Done! Check `{OUTPUT_DIR}/` for your MP3 files.")
