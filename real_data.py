"""
SellerShield — real_data.py

FINAL SCRAPING STRATEGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Platform   │ Method                   │ Library
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Amazon     │ requests scraping        │ requests
Flipkart   │ requests scraping        │ requests
eBay       │ requests scraping        │ requests
Etsy       │ requests scraping        │ requests
Meesho     │ Selenium (full browser)  │ selenium + chrome
Myntra     │ Selenium (full browser)  │ selenium + chrome
Shopsy     │ requests scraping        │ requests
Snapdeal   │ requests scraping        │ requests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All platforms fall back to synthetic ML if scraping fails.

INSTALL REQUIREMENTS:
    pip install selenium webdriver-manager
"""

import requests
import re
import math
import time
import hashlib
import json

# ── Selenium imports (graceful if not installed) ───────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        WEBDRIVER_MANAGER = True
    except ImportError:
        WEBDRIVER_MANAGER = False
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed. Run: pip install selenium webdriver-manager")
    print("   Meesho and Myntra will use synthetic fallback until Selenium is installed.")

TIMEOUT    = 12
SELENIUM_TIMEOUT = 20  # Selenium needs more time to load JS

# ── Browser-like headers for requests ─────────────────────────────────
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control":   "max-age=0",
}

# Platform baselines used for synthetic fallback
_PROFILES = {
    "Amazon":   {"age":36, "reviews":450, "rating":4.2, "fraud":0.12, "resp":12,  "lq":72},
    "Flipkart": {"age":28, "reviews":320, "rating":4.1, "fraud":0.15, "resp":18,  "lq":65},
    "eBay":     {"age":48, "reviews":680, "rating":4.3, "fraud":0.18, "resp":10,  "lq":80},
    "Etsy":     {"age":42, "reviews":290, "rating":4.6, "fraud":0.10, "resp":8,   "lq":82},
    "Meesho":   {"age":10, "reviews":75,  "rating":3.9, "fraud":0.28, "resp":30,  "lq":48},
    "Myntra":   {"age":30, "reviews":380, "rating":4.2, "fraud":0.13, "resp":15,  "lq":70},
    "Shopsy":   {"age":8,  "reviews":55,  "rating":3.8, "fraud":0.30, "resp":36,  "lq":44},
    "Snapdeal": {"age":22, "reviews":175, "rating":3.8, "fraud":0.22, "resp":24,  "lq":55},
}


# ══════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════

def fetch_real_data(seller_name, platform):
    """
    Routes each platform to the right scraper.
    Always returns: (features_dict, source_label)
    """
    print(f"\n🔍 [{platform}] Scraping: '{seller_name}'")

    scrapers = {
        "Amazon":   _scrape_amazon,
        "Flipkart": _scrape_flipkart,
        "eBay":     _scrape_ebay,
        "Etsy":     _scrape_etsy,
        "Meesho":   _scrape_meesho_selenium,   # ← Full browser
        "Myntra":   _scrape_myntra_selenium,   # ← Full browser
        "Shopsy":   _scrape_shopsy,
        "Snapdeal": _scrape_snapdeal,
    }

    scraper = scrapers.get(platform)
    if scraper:
        try:
            result = scraper(seller_name)
            if result:
                return result
        except Exception as e:
            print(f"   ❌ {platform} scraper error: {e}")

    return _synthetic(seller_name, platform)


# ══════════════════════════════════════════════════════════════
# SELENIUM BROWSER HELPER
# ══════════════════════════════════════════════════════════════

def _get_selenium_driver():
    """
    Creates a headless Chrome browser instance.
    Configured to look like a real browser to avoid detection.
    """
    if not SELENIUM_AVAILABLE:
        return None

    options = Options()

    # Run headless (no visible window)
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument(f"--user-agent={HEADERS['User-Agent']}")

    # Disable automation flags that websites detect
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Disable images for faster loading
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    try:
        if WEBDRIVER_MANAGER:
            # Auto-download correct ChromeDriver version
            service = Service(ChromeDriverManager().install())
            driver  = webdriver.Chrome(service=service, options=options)
        else:
            # Use system ChromeDriver
            driver = webdriver.Chrome(options=options)

        # Remove webdriver property to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver

    except Exception as e:
        print(f"   ❌ Chrome driver error: {e}")
        print("   💡 Make sure Chrome is installed: https://www.google.com/chrome/")
        return None


# ══════════════════════════════════════════════════════════════
# MEESHO — Selenium full browser scraping
# ══════════════════════════════════════════════════════════════

def _scrape_meesho_selenium(seller_name):
    """
    Meesho is React JS — normal requests gets empty page.
    Uses Selenium to fully load JavaScript then extract data.
    """
    if not SELENIUM_AVAILABLE:
        print("   ⚠️  Selenium not installed → synthetic fallback")
        print("   📌 Install: pip install selenium webdriver-manager")
        return None

    driver = _get_selenium_driver()
    if not driver:
        return None

    try:
        search_url = f"https://meesho.com/search?q={requests.utils.quote(seller_name)}"
        print(f"   🌐 Opening Meesho in browser: {search_url}")

        driver.get(search_url)

        # Wait for products to load
        try:
            WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                    "[class*='ProductCard'], [class*='product-card'], [data-testid*='product']"))
            )
        except TimeoutException:
            print("   ⚠️  Meesho: Products didn't load in time")

        # Extra wait for JS to finish rendering
        time.sleep(3)

        html = driver.page_source

        # ── Try to extract ratings from loaded page ────────────────────
        ratings  = []
        counts   = []

        # Method 1 — find rating elements by CSS
        try:
            rating_elements = driver.find_elements(By.CSS_SELECTOR,
                "[class*='rating'], [class*='Rating'], [class*='star'], [data-rating]")
            for el in rating_elements[:20]:
                text = el.text.strip()
                m = re.match(r'^(\d+\.?\d*)$', text)
                if m:
                    val = float(m.group(1))
                    if 1.0 <= val <= 5.0:
                        ratings.append(val)
        except: pass

        # Method 2 — parse JSON from page source
        try:
            json_blobs = re.findall(r'"averageRating":\s*(\d+\.?\d*)', html)
            for v in json_blobs:
                val = float(v)
                if 1.0 <= val <= 5.0:
                    ratings.append(val)
        except: pass

        # Method 3 — React props JSON
        try:
            props_match = re.findall(r'"rating":\s*"?(\d+\.?\d*)"?', html)
            for v in props_match:
                val = float(v)
                if 1.0 <= val <= 5.0:
                    ratings.append(val)
        except: pass

        # Extract counts
        try:
            count_els = driver.find_elements(By.CSS_SELECTOR,
                "[class*='review-count'], [class*='ratingCount'], [class*='rating-count']")
            for el in count_els[:10]:
                text = re.sub(r'[^\d]', '', el.text)
                if text:
                    counts.append(int(text))
        except: pass

        try:
            count_matches = re.findall(r'"ratingCount":\s*(\d+)', html)
            for v in count_matches:
                counts.append(int(v))
        except: pass

        # ── Also try to get seller-specific info ───────────────────────
        try:
            seller_elements = driver.find_elements(By.CSS_SELECTOR,
                "[class*='supplier'], [class*='seller'], [class*='brand']")
            for el in seller_elements[:5]:
                if seller_name.lower() in el.text.lower():
                    print(f"   ✅ Found seller mention: {el.text[:50]}")
                    break
        except: pass

        driver.quit()

        if ratings and counts:
            avg_rating    = round(sum(ratings) / len(ratings), 2)
            total_reviews = max(counts)
            print(f"   ✅ Meesho Selenium — {total_reviews:,} reviews, {avg_rating}★")
            return _build(avg_rating, total_reviews, 0, "Meesho", "meesho_selenium")

        # Got page but no structured data
        if len(html) > 10000:
            print("   ⚠️  Meesho loaded but no rating data found → smart parse")
            rating  = _extract_rating(html, [r'"rating":\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*★'])
            reviews = _extract_count(html,  [r'"ratingCount":\s*(\d+)', r'([\d,]+)\s*[Rr]atings?'])
            if rating and reviews:
                print(f"   ✅ Meesho parsed — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, 0, "Meesho", "meesho_selenium")

        print("   ⚠️  Meesho: No data extracted → synthetic fallback")
        return None

    except Exception as e:
        print(f"   ❌ Meesho Selenium error: {e}")
        try: driver.quit()
        except: pass
        return None


# ══════════════════════════════════════════════════════════════
# MYNTRA — Selenium full browser scraping
# ══════════════════════════════════════════════════════════════

def _scrape_myntra_selenium(seller_name):
    """
    Myntra is Next.js — loads content via client-side JS.
    Uses Selenium to fully render the page then extract data.
    """
    if not SELENIUM_AVAILABLE:
        print("   ⚠️  Selenium not installed → synthetic fallback")
        print("   📌 Install: pip install selenium webdriver-manager")
        return None

    driver = _get_selenium_driver()
    if not driver:
        return None

    try:
        search_url = f"https://www.myntra.com/{requests.utils.quote(seller_name.lower().replace(' ', '-'))}"
        print(f"   🌐 Opening Myntra in browser: {search_url}")

        driver.get(search_url)

        # Wait for products grid to appear
        try:
            WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                    ".product-base, .product-productMetaInfo, [class*='product-base']"))
            )
        except TimeoutException:
            # Try search page instead
            search_url2 = f"https://www.myntra.com/search?rawQuery={requests.utils.quote(seller_name)}"
            print(f"   🔄 Trying search URL: {search_url2}")
            driver.get(search_url2)
            try:
                WebDriverWait(driver, SELENIUM_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                        ".product-base, [class*='results-base']"))
                )
            except TimeoutException:
                print("   ⚠️  Myntra: Page didn't load in time")

        time.sleep(3)

        html = driver.page_source

        # ── Method 1: Extract from __NEXT_DATA__ JSON ──────────────────
        next_match = re.search(r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if next_match:
            try:
                data     = json.loads(next_match.group(1))
                products = (data.get("props", {})
                                .get("pageProps", {})
                                .get("initialState", {})
                                .get("products", {})
                                .get("products", []))
                ratings  = [float(p["rating"])      for p in products if p.get("rating")]
                counts   = [int(p["ratingCount"])    for p in products if p.get("ratingCount")]
                if ratings and counts:
                    avg_rating    = round(sum(ratings)/len(ratings), 2)
                    total_reviews = max(counts)
                    driver.quit()
                    print(f"   ✅ Myntra Selenium (NEXT_DATA) — {total_reviews:,} reviews, {avg_rating}★")
                    return _build(avg_rating, total_reviews, 0, "Myntra", "myntra_selenium")
            except Exception as e:
                print(f"   ⚠️  NEXT_DATA parse failed: {e}")

        # ── Method 2: Find rating elements directly ────────────────────
        ratings = []
        counts  = []

        try:
            # Myntra product rating elements
            rating_elements = driver.find_elements(By.CSS_SELECTOR,
                ".product-ratingsContainer .product-ratingsCount, "
                "[class*='product-ratingsCount'], "
                ".index-overallRating, "
                "[class*='overallRating']")
            for el in rating_elements[:20]:
                text = el.text.strip()
                m = re.match(r'^(\d+\.?\d*)$', text)
                if m:
                    val = float(m.group(1))
                    if 1.0 <= val <= 5.0:
                        ratings.append(val)
        except: pass

        try:
            count_elements = driver.find_elements(By.CSS_SELECTOR,
                "[class*='ratingCount'], [class*='rating-count'], .product-ratingsCount")
            for el in count_elements[:10]:
                text = re.sub(r'[^\d]', '', el.text)
                if text:
                    counts.append(int(text))
        except: pass

        # ── Method 3: Parse JSON from page source ──────────────────────
        rating  = _extract_rating(html, [
            r'"averageRating":\s*(\d+\.?\d*)',
            r'"rating":\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*★',
        ])
        reviews_count = _extract_count(html, [
            r'"ratingCount":\s*(\d+)',
            r'([\d,]+)\s*[Rr]atings?',
            r'([\d,]+)\s*[Rr]eviews?',
        ])

        if rating:   ratings.append(rating)
        if reviews_count: counts.append(reviews_count)

        driver.quit()

        if ratings and counts:
            avg_rating    = round(sum(ratings)/len(ratings), 2)
            total_reviews = max(counts)
            print(f"   ✅ Myntra Selenium — {total_reviews:,} reviews, {avg_rating}★")
            return _build(avg_rating, total_reviews, 0, "Myntra", "myntra_selenium")

        print("   ⚠️  Myntra: No data extracted → synthetic fallback")
        return None

    except Exception as e:
        print(f"   ❌ Myntra Selenium error: {e}")
        try: driver.quit()
        except: pass
        return None


# ══════════════════════════════════════════════════════════════
# AMAZON — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_amazon(seller_name):
    session = _new_session()
    urls = [
        f"https://www.amazon.in/s?me={requests.utils.quote(seller_name)}",
        f"https://www.amazon.in/s?k={requests.utils.quote(seller_name)}",
    ]
    for url in urls:
        try:
            session.get("https://www.amazon.in", timeout=8)
            time.sleep(0.5)
            r    = session.get(url, timeout=TIMEOUT)
            html = r.text
            if _is_blocked(html, ["robot check", "captcha", "automated access"]): continue
            rating  = _extract_rating(html, [
                r'(\d+\.?\d*)\s*out of\s*5\s*stars',
                r'"averageRating":\s*"?(\d+\.?\d*)"?',
            ])
            reviews = _extract_count(html, [
                r'([\d,]+)\s*(?:global\s*)?ratings?',
                r'"ratingCount":\s*(\d+)',
            ])
            if rating and reviews:
                verified = "Amazon's Choice" in html or "fulfilled by amazon" in html.lower()
                print(f"   ✅ Amazon scraped — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, int(verified), "Amazon", "amazon_scrape")
        except Exception as e:
            print(f"   ⚠️  Amazon URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# FLIPKART — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_flipkart(seller_name):
    session = _new_session()
    urls = [
        f"https://www.flipkart.com/search?q={requests.utils.quote(seller_name)}",
        f"https://www.flipkart.com/search?q={requests.utils.quote(seller_name)}&otracker=search",
    ]
    for url in urls:
        try:
            r    = session.get(url, timeout=TIMEOUT, headers={**HEADERS, "Referer":"https://www.flipkart.com/"})
            html = r.text
            if _is_blocked(html, ["captcha"]) or len(html) < 3000: continue
            rating  = _extract_rating(html, [
                r'"averageRating":\s*"?(\d+\.?\d*)"?',
                r'"rating":\s*"(\d+\.?\d*)"',
                r'(\d+\.?\d*)\s*★',
            ])
            reviews = _extract_count(html, [
                r'"ratingCount":\s*(\d+)',
                r'"totalRatings":\s*(\d+)',
                r'([\d,]+)\s*[Rr]atings?',
            ])
            if rating and reviews:
                verified = '"isFlipkartAssured":true' in html or "Flipkart Assured" in html
                print(f"   ✅ Flipkart scraped — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, int(verified), "Flipkart", "flipkart_scrape")
        except Exception as e:
            print(f"   ⚠️  Flipkart URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# eBay — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_ebay(seller_name):
    session = _new_session()
    urls = [
        f"https://www.ebay.com/usr/{requests.utils.quote(seller_name)}",
        f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(seller_name)}&_ssn={requests.utils.quote(seller_name)}",
        f"https://www.ebay.com/sch/i.html?_nkw={requests.utils.quote(seller_name)}",
    ]
    for url in urls:
        try:
            r    = session.get(url, timeout=TIMEOUT)
            html = r.text
            if len(html) < 2000: continue
            if "/usr/" in url:
                fb_pct  = _extract_rating(html, [r'(\d+\.?\d*)%\s*[Pp]ositive', r'"feedbackPercentage":\s*"?(\d+\.?\d*)"?'])
                rating  = round(fb_pct / 100 * 5, 2) if fb_pct else None
                reviews = _extract_count(html, [r'"feedbackCount":\s*(\d+)', r'([\d,]+)\s*[Ff]eedback'])
            else:
                rating  = _extract_rating(html, [r'(\d+\.?\d*)\s*out of 5\s*stars', r'"averageRating":\s*"?(\d+\.?\d*)"?'])
                reviews = _extract_count(html, [r'([\d,]+)\s*product ratings?', r'"reviewCount":\s*(\d+)'])
            if rating and reviews and 1.0 <= rating <= 5.0:
                verified = "eBay Top Rated" in html or "top-rated" in html.lower()
                print(f"   ✅ eBay scraped — {reviews:,} feedback, {rating}★")
                return _build(rating, reviews, int(verified), "eBay", "ebay_scrape")
        except Exception as e:
            print(f"   ⚠️  eBay URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# Etsy — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_etsy(seller_name):
    session = _new_session()
    urls = [
        f"https://www.etsy.com/shop/{requests.utils.quote(seller_name)}",
        f"https://www.etsy.com/search/shops?q={requests.utils.quote(seller_name)}",
    ]
    for url in urls:
        try:
            r    = session.get(url, timeout=TIMEOUT, headers={**HEADERS, "Referer":"https://www.etsy.com/"})
            html = r.text
            if len(html) < 2000 or "Page not found" in html: continue
            # JSON-LD
            for blob in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL):
                try:
                    data = json.loads(blob)
                    if isinstance(data, list): data = data[0]
                    agg = data.get("aggregateRating", {})
                    if agg:
                        rating  = float(agg.get("ratingValue", 0))
                        reviews = int(agg.get("reviewCount", 0))
                        if 1.0 <= rating <= 5.0 and reviews > 0:
                            verified = "star seller" in html.lower()
                            print(f"   ✅ Etsy scraped — {reviews:,} reviews, {rating}★")
                            return _build(rating, reviews, int(verified), "Etsy", "etsy_scrape")
                except: continue
            rating  = _extract_rating(html, [r'"ratingValue":\s*"?(\d+\.?\d*)"?', r'(\d+\.?\d*)\s*out of 5'])
            reviews = _extract_count(html,  [r'"reviewCount":\s*(\d+)', r'([\d,]+)\s*[Rr]eviews?'])
            if rating and reviews:
                print(f"   ✅ Etsy scraped — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, 0, "Etsy", "etsy_scrape")
        except Exception as e:
            print(f"   ⚠️  Etsy URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# Shopsy — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_shopsy(seller_name):
    session = _new_session()
    urls = [
        f"https://www.shopsy.in/search?q={requests.utils.quote(seller_name)}",
        f"https://www.flipkart.com/search?q={requests.utils.quote(seller_name)}&marketplace=SHOPSY",
    ]
    for url in urls:
        try:
            r    = session.get(url, timeout=TIMEOUT, headers={**HEADERS, "Referer":"https://www.shopsy.in/"})
            html = r.text
            if len(html) < 2000: continue
            rating  = _extract_rating(html, [r'"averageRating":\s*"?(\d+\.?\d*)"?', r'"rating":\s*"(\d+\.?\d*)"'])
            reviews = _extract_count(html,  [r'"ratingCount":\s*(\d+)', r'([\d,]+)\s*[Rr]atings?'])
            if rating and reviews:
                print(f"   ✅ Shopsy scraped — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, 0, "Shopsy", "shopsy_scrape")
        except Exception as e:
            print(f"   ⚠️  Shopsy URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# Snapdeal — requests scraping
# ══════════════════════════════════════════════════════════════

def _scrape_snapdeal(seller_name):
    session = _new_session()
    urls = [
        f"https://www.snapdeal.com/search?keyword={requests.utils.quote(seller_name)}",
    ]
    for url in urls:
        try:
            r    = session.get(url, timeout=TIMEOUT, headers={**HEADERS, "Referer":"https://www.snapdeal.com/"})
            html = r.text
            if len(html) < 2000 or _is_blocked(html, ["captcha"]): continue
            rating  = _extract_rating(html, [r'"averageRating":\s*"?(\d+\.?\d*)"?', r'(\d+\.?\d*)\s*out of 5'])
            reviews = _extract_count(html,  [r'"ratingCount":\s*(\d+)', r'([\d,]+)\s*[Rr]atings?', r'"numRatings":\s*(\d+)'])
            if rating and reviews:
                verified = "snapdeal certified" in html.lower()
                print(f"   ✅ Snapdeal scraped — {reviews:,} reviews, {rating}★")
                return _build(rating, reviews, int(verified), "Snapdeal", "snapdeal_scrape")
        except Exception as e:
            print(f"   ⚠️  Snapdeal URL error: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK
# ══════════════════════════════════════════════════════════════

def _synthetic(seller_name, platform):
    name = seller_name.lower()
    RED    = ["free","win","prize","urgent","cheap","fake","wholesale","clearance","loot","limited"]
    YELLOW = ["deal","offer","sale","best","official","verified","trusted","original","genuine"]
    GREEN  = ["store","shop","brand","mart","co","ltd","inc","enterprise","traders","exports"]

    red_c  = sum(1 for w in RED    if w in name)
    yel_c  = sum(1 for w in YELLOW if w in name)
    grn_c  = sum(1 for w in GREEN  if w in name)
    sp_c   = len(re.findall(r'[0-9!@#$%]', seller_name))

    risk = 1.0
    risk -= red_c * 0.15
    risk -= yel_c * 0.05
    risk += grn_c * 0.08
    risk *= max(0.3, 1.0 - sp_c * 0.1)
    risk *= min(1.0, len(seller_name) / 14)
    risk  = max(0.2, min(1.3, risk))

    h = int(hashlib.md5((name + platform).encode()).hexdigest(), 16) % 100000
    def vary(base, var, seed):
        return base * (1.0 + math.sin(seed * (h % 9973 + 1)) * var)

    p = _PROFILES.get(platform, {"age":24,"reviews":200,"rating":4.0,"fraud":0.20,"resp":20,"lq":60})

    features = {
        "account_age_months":  round(max(1,   vary(p["age"],     0.35, 1) * risk), 1),
        "total_reviews":       max(0, int(    vary(p["reviews"], 0.45, 2) * risk)),
        "avg_rating":          round(min(5.0, max(1.0, vary(p["rating"], 0.06, 3))), 2),
        "rating_std":          round(max(0.05,vary(0.5,          0.40, 4) * (2-risk)), 2),
        "return_rate":         round(min(0.80,max(0.01, p["fraud"]*(2-risk)*vary(1,0.3,5))), 3),
        "response_time_hrs":   round(max(1,   vary(p["resp"],    0.40, 6) * (2-risk)), 1),
        "price_deviation_pct": round(         vary(-5,           2.50, 7) * (2-risk), 1),
        "platform_verified":   1 if risk > 0.9 and h % 4 == 0 else 0,
        "listing_quality":     round(min(100, max(10, vary(p["lq"], 0.25, 8) * risk)), 1),
        "dispute_rate":        round(min(0.80,max(0.01, p["fraud"]*(2-risk)*vary(1,0.3,9))), 3),
        "repeat_buyer_rate":   round(min(0.90,max(0.05, vary(0.38, 0.30, 10)*risk)), 3),
    }
    print(f"   📊 Synthetic fallback — risk={risk:.2f}, reviews={features['total_reviews']}, rating={features['avg_rating']}")
    return features, "synthetic_ml"


# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

def _new_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def _is_blocked(html, keywords):
    return any(k in html.lower() for k in keywords)

def _extract_rating(html, patterns):
    for pattern in patterns:
        for m in re.findall(pattern, html, re.IGNORECASE):
            try:
                val = float(m)
                if 1.0 <= val <= 5.0:
                    return val
            except: continue
    return None

def _extract_count(html, patterns):
    for pattern in patterns:
        for m in re.findall(pattern, html, re.IGNORECASE):
            try:
                val = int(str(m).replace(',','').strip())
                if val > 0:
                    return val
            except: continue
    return None

def _build(rating, reviews, verified, platform, source):
    p            = _PROFILES.get(platform, {})
    fraud_rate   = p.get("fraud", 0.15)
    rating_std   = 0.12 if rating > 4.8 else (0.35 if rating > 4.2 else 0.65)
    dispute_rate = round(max(0.01, (5 - rating) / 5 * fraud_rate * 2), 3)
    repeat_rate  = round(min(0.90, max(0.10, (rating - 1) / 4 * 0.7)), 3)
    return {
        "account_age_months":  p.get("age",  24),
        "total_reviews":       reviews,
        "avg_rating":          round(rating, 2),
        "rating_std":          rating_std,
        "return_rate":         round(fraud_rate * (2 - rating/5), 3),
        "response_time_hrs":   p.get("resp", 20),
        "price_deviation_pct": 0,
        "platform_verified":   verified,
        "listing_quality":     p.get("lq",   60),
        "dispute_rate":        dispute_rate,
        "repeat_buyer_rate":   repeat_rate,
    }, source
