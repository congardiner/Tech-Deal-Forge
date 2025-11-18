import os
import re
import time
from datetime import datetime
from botasaurus.browser import browser, Driver
from botasaurus.soupify import soupify
from data_pipeline import DealsDataPipeline
from pathlib import Path

# Best Buy Deal URLs


BESTBUY_URLS = [
    "https://www.bestbuy.com/site/promo/black-friday-laptop-computer-deals-1",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1591132221892&id=pcat17071&qp=currentoffers_facet=Current+Deals%7EOn+Sale&st=pcmcat1591132221892_categoryid%24abcat0500000"
]


HEADLESS = os.getenv('BESTBUY_HEADLESS', 'true').lower() == 'true'
WAIT_FOR_COMPLETE_PAGE = os.getenv('BESTBUY_WAIT_FOR_COMPLETE', 'true').lower() == 'true'
SCROLL_COUNT = int(os.getenv('BESTBUY_SCROLL_COUNT', '5'))
WAIT_AFTER_SCROLL = float(os.getenv('BESTBUY_WAIT_AFTER_SCROLL', '3'))

def extract_price_numeric(price_text):
    """Extract numeric value from price string like '$1,299.99' -> 1299.99"""
    if not price_text:
        return None
    # Keep decimals, remove commas and currency
    cleaned = re.sub(r"[^\d\.]", "", str(price_text))
    try:
        return float(cleaned) if cleaned else None
    except Exception:
        return None

def calculate_discount(original, current):
    """Calculate discount percentage"""
    try:
        orig = extract_price_numeric(original)
        curr = extract_price_numeric(current)
        if orig and curr and orig > curr:
            return round(((orig - curr) / orig) * 100, 2)
    except Exception:
        pass
    return None

@browser(
    headless=HEADLESS, 
    reuse_driver=True, 
    block_images=False,  # Allow images to ensure full page render
    wait_for_complete_page_load=WAIT_FOR_COMPLETE_PAGE,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
def scrape_bestbuy_deals(driver: Driver, url: str):
    """Scrape a Best Buy category/listing URL and return a list of deal dicts.
    Uses botasaurus features for optimal scraping.
    """
    print(f"Loading URL: {url}")
    driver.get(url)
    
    # Wait for initial page load (Best Buy is React-heavy)
    print("Waiting for page to render...")
    driver.wait_for_element("body", wait=10)
    
    # Give React time to hydrate
    time.sleep(3)
    
    # Scroll to trigger lazy-loaded content
    print(f"Scrolling page {SCROLL_COUNT} times to load all products...")
    for i in range(SCROLL_COUNT):
        driver.scroll_to_bottom()
        time.sleep(WAIT_AFTER_SCROLL)
        print(f"  Scroll {i+1}/{SCROLL_COUNT} complete")
    
    # Final wait for any remaining content
    time.sleep(2)
    
    soup = soupify(driver.page_html)

    # Basic bot/blocked detection
    page_text = soup.get_text(" ", strip=True).lower()
    if any(bad in page_text for bad in ["access denied", "verify you are a human", "robot check"]):
        print("⚠️ BestBuy blocked or human verification required. Try slower runs or a different network.")
        return []

    deals = []

    # Try multiple robust product card selectors (expanded for new DOMs)
    # Best Buy uses very specific class names that change with their React builds
    selectors = [
        # Current Best Buy selectors (as of 2025)
        "div.sku-item",
        "li.sku-item",
        "ol.sku-item-list li",
        "li[class*='sku-item']",
        "div[class*='sku-item']",
        "div[class*='shop-sku-list'] li",
        "li[class*='shop-sku-list-item']",
        "div[class*='product'] article",
        "div[class*='product-grid'] li",
        # React-specific containers
        "div[data-testid*='product']",
        "div[class*='ProductCard']",
        "article[class*='product']",
    ]

    product_cards = []
    if soup:
        for sel in selectors:
            product_cards = soup.select(sel)
            if product_cards:
                print(f"✓ Found {len(product_cards)} products using selector: {sel}")
                break

    if not product_cards and soup:
        # Fallback: look for cards with data-sku-id
        product_cards = soup.select("[data-sku-id]")
        if product_cards:
            print(f"✓ Found {len(product_cards)} products using data-sku-id fallback")

    if not product_cards:
        # Persist the HTML for debugging
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        err_dir = Path(__file__).resolve().parent / "error_logs" / ts
        try:
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_html)
            print(f"⚠️ No product cards found. Saved HTML to {err_dir / 'page.html'}")
            print(f"   Page length: {len(driver.page_html)} chars")
            print(f"   Try running with BESTBUY_HEADLESS=false to see what's happening")
        except Exception as e:
            print(f"   Failed to save debug HTML: {e}")

    # Detect category from URL
    category = 'Tech Deals'
    url_l = url.lower()
    if 'gaming' in url_l:
        category = 'Gaming'
    elif 'laptop' in url_l:
        category = 'Laptops'
    elif 'monitor' in url_l:
        category = 'Monitors'
    elif 'tv' in url_l:
        category = 'TVs'
    elif 'tablet' in url_l or 'ipad' in url_l:
        category = 'Tablets'
    elif 'nintendo' in url_l:
        category = 'Nintendo'
    elif 'headphone' in url_l or 'audio' in url_l:
        category = 'Headphones'

    for card in product_cards:
        # Title + link
        title_elem = (
            card.select_one('.sku-title a')
            or card.select_one('.sku-header a')
            or card.select_one('h4 a')
            or card.select_one("a[href^='/site/']")
            or card.find('a', href=True)
        )
        title = title_elem.get_text(strip=True) if title_elem else None
        href = title_elem.get('href') if title_elem else None
        link = (
            href if href and href.startswith('http')
            else f"https://www.bestbuy.com{href}" if href else None
        )

        # Fallback: build product link from SKU id if link is missing or not a product URL
        if not link or "bestbuy.com" not in link:
            sku = card.get('data-sku-id')
            if not sku:
                # Try to find nested element with data-sku-id
                sku_holder = card.select_one('[data-sku-id]')
                if sku_holder:
                    sku = sku_holder.get('data-sku-id')
            if sku and sku.isdigit():
                # Generic SKU link that BestBuy redirects to the correct product page
                link = f"https://www.bestbuy.com/site/{sku}.p?skuId={sku}"

        # Price (try several selectors, then regex fallback)
        price_text = None
        price_selectors = [
            ".priceView-customer-price span[aria-hidden='true']",
            ".priceView-hero-price span[aria-hidden='true']",
            "[data-testid='customer-price'] span",
            "div[class*='pricing-price__regular-price']",
            "div[class*='pricing-price'] span",
            "div[class*='price'] span",
        ]
        for sel in price_selectors:
            el = card.select_one(sel)
            if el:
                txt = el.get_text(strip=True)
                if "$" in txt:
                    price_text = txt
                    break
        if not price_text:
            # Fallback: any $xx.xx in the card
            text = card.get_text(" ", strip=True)
            m = re.search(r"\$\s*\d+[\d,]*(?:\.\d{2})?", text)
            price_text = m.group(0) if m else None

        # Original price (crossed out / was price)
        original_price_text = None
        original_selectors = [
            "[class*='regular-price']",
            "[class*='was-price']",
            "s[class*='price']",
            "[data-testid='regular-price']",
        ]
        for sel in original_selectors:
            el = card.select_one(sel)
            if el:
                otxt = el.get_text(strip=True)
                if "$" in otxt or "Comp" in otxt or "Was" in otxt:
                    original_price_text = otxt
                    break

        # Rating + reviews
        rating_val = None
        reviews_count = None
        rating_block = card.select_one("[class*='review']") or card.select_one(".c-reviews-v4")
        if rating_block:
            text = rating_block.get_text(" ", strip=True)
            rm = re.search(r"(\d+\.?\d*)\s*out of", text, re.IGNORECASE)
            if rm:
                try:
                    rating_val = float(rm.group(1))
                except Exception:
                    rating_val = None
            cm = re.search(r"\(([^\)]+)\)", text)
            if cm:
                digits = re.sub(r"[^\d]", "", cm.group(1))
                if digits:
                    try:
                        reviews_count = int(digits)
                    except Exception:
                        reviews_count = None

        # Build deal
        if title and link and price_text:
            price_num = extract_price_numeric(price_text)
            discount_percent = calculate_discount(original_price_text, price_text)
            deals.append({
                'title': title,
                'link': link,
                'price': price_text,
                'original_price': original_price_text,
                'discount_percent': discount_percent,
                'rating': rating_val,
                'reviews_count': reviews_count,
                'category': category,
                'website': 'bestbuy',
                'price_numeric': price_num,
                'price_text': price_text,
                'scraped_at': datetime.now().isoformat()
            })

    return deals

def main():
    print("Best Buy Tech Deals Scraper")
    print("-" * 60)
    
    all_deals = []
    successful_urls = 0
    failed_urls = []
    
    for url in BESTBUY_URLS:
        try:
            deals = scrape_bestbuy_deals(url)
            if deals:
                all_deals.extend(deals)
                successful_urls += 1
                print(f"Scraped {len(deals)} deals from URL {successful_urls}")
            else:
                failed_urls.append(url[:80])
        except Exception as e:
            print(f"Error scraping URL: {str(e)[:50]}")
            failed_urls.append(url[:80])
    
    print("-" * 60)
    print(f"Summary: {successful_urls}/{len(BESTBUY_URLS)} URLs successful")
    print(f"Total deals: {len(all_deals)}")
    print(f"Deals with prices: {len([d for d in all_deals if d.get('price')])}")
    
    if failed_urls:
        print(f"Failed URLs: {len(failed_urls)}")
    
    if not all_deals:
        print("Warning: No deals collected")
        return
    
    # Use project-root output folder to match Streamlit's DB path
    project_root = Path(__file__).resolve().parent
    pipeline = DealsDataPipeline(output_dir=str(project_root / "output"), use_mysql=False)
    result = pipeline.process_deals(all_deals, csv_prefix="bestbuy")
    
    print(f"Saved to: {result['csv']}")
    print(f"Database entries added: {result['database_rows_added']}")
    print("Scraping complete")
    print("-" * 60)

if __name__ == "__main__":
    main()

