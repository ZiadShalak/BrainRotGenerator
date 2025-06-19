#!/usr/bin/env python3
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
BOARD_URL     = "https://www.pinterest.com/thebluffington/reaction-folder/"
OUTPUT_DIR    = "images"
SCROLL_PAUSE  = 3.0              # seconds to wait after each scroll
MAX_SCROLLS   = 200              # max viewport-height scrolls
MIN_WIDTH_PX  = 200              # only grab images at least this wide
EXTENSIONS    = (".png", ".jpg", ".jpeg")

# ─── PREPARE OUTPUT ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LAUNCH CHROME ───────────────────────────────────────────────────────────────
opts = Options()
opts.add_argument("--headless")
opts.add_argument("--disable-gpu")
opts.add_argument("--window-size=1920,1080")
# pretend to be a normal browser
opts.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.5735.133 Safari/537.36"
)
# anti-automation
opts.add_experimental_option("excludeSwitches", ["enable-automation"])
opts.add_experimental_option("useAutomationExtension", False)
opts.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=opts
)

# ─── NAVIGATE & SCROLL ───────────────────────────────────────────────────────────
driver.get(BOARD_URL)
time.sleep(5)  # initial load

seen_urls = set()
prev_count = 0

for i in range(1, MAX_SCROLLS + 1):
    driver.execute_script("window.scrollBy(0, window.innerHeight);")
    time.sleep(SCROLL_PAUSE)

    # grab fresh list each pass
    imgs = driver.find_elements(By.TAG_NAME, "img")
    for img in imgs:
        try:
            # try srcset first for highest-res
            srcset = img.get_attribute("srcset") or ""
            url = None
            if srcset:
                candidates = []
                for part in srcset.split(","):
                    u_w = part.strip().rsplit(" ", 1)
                    if len(u_w) != 2:
                        continue
                    u, w = u_w
                    if not w.endswith("w"):
                        continue
                    try:
                        width_px = int(w[:-1])
                    except ValueError:
                        continue
                    if u.lower().endswith(EXTENSIONS):
                        candidates.append((width_px, u))
                if candidates:
                    _, url = max(candidates)

            # fallback to src
            if not url:
                src = img.get_attribute("src") or ""
                if src.lower().endswith(EXTENSIONS):
                    url = src

            # skip if still nothing or too small
            if not url:
                continue
            natural_w = img.get_property("naturalWidth") or 0
            if natural_w < MIN_WIDTH_PX:
                continue

            seen_urls.add(url)

        except StaleElementReferenceException:
            # element went stale—skip it
            continue

    # stop if no new images this scroll
    if len(seen_urls) == prev_count:
        print(f"No new images after {i} scrolls; stopping early.")
        break
    prev_count = len(seen_urls)
    print(f"Scroll {i}/{MAX_SCROLLS}: found {prev_count} unique images so far…")

driver.quit()

print(f"\n✅ Total unique HD images found: {len(seen_urls)}. Starting download…\n")

# ─── DOWNLOAD ────────────────────────────────────────────────────────────────────
for idx, url in enumerate(sorted(seen_urls), start=1):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        ext = os.path.splitext(url.split("?", 1)[0])[1]
        filename = f"meme_{idx:03}{ext}"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"[{idx}/{len(seen_urls)}] Saved {filename}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

print(f"\n🎉 Done! {len(seen_urls)} images in `{OUTPUT_DIR}/`.")
