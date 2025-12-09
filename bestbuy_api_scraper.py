import os
import re
import json
import time
from datetime import datetime
from botasaurus.browser import browser, Driver
from data_pipeline import DealsDataPipeline
from pathlib import Path

# NOTE: (REGEX) RE was added to clarify the purpose of the following list, as it was not immediately clear and to assign and match patterns consistently.
# NOTE: Best Buy Deal URLs (ie, tested and validated over the span of my project) (some of these may be seasonal or change over time though overtime)
BESTBUY_URLS = [
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1591132221892&id=pcat17071&qp=currentoffers_facet=Current+Deals%7EOn+Sale&st=pcmcat1591132221892_categoryid%24abcat0500000",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1591132221892&id=pcat17071&qp=currentoffers_facet%3DCurrent+Deals%7EBlack+Friday+Deal&st=pcmcat1591132221892_categoryid%24abcat0500000",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1720706651516&id=pcat17071&qp=currentoffers_facet%3DSeasonal+Savings%7EBlack+Friday+Deal&st=pcmcat1720706651516_categoryid%24pcmcat156400050037",
    "https://www.bestbuy.com/site/pc-gaming/gaming-laptops/pcmcat287600050003.c?id=pcmcat287600050003",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat138500050001&id=pcat17071&qp=currentoffers_facet%3DCurrent+Deals%7EOn+Sale%5Econdition_facet%3DCondition%7ENew&st=categoryid%24pcmcat138500050001",
    "https://www.bestbuy.com/site/xbox-series-x-and-s/xbox-series-x-and-s-consoles/pcmcat1586900952752.c?id=pcmcat1586900952752",
    "https://www.bestbuy.com/site/all-electronics-on-sale/all-wearable-technology-on-sale/pcmcat1690897435393.c?id=pcmcat1690897435393",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1506545802590&id=pcat17071&qp=headphonefit_facet%3DHeadphone+Fit%7EOver-the-Ear&st=pcmcat1506545802590_categoryid%24pcmcat144700050004", # Over-Ear Headphones
    "https://www.bestbuy.com/site/all-black-friday-deals/black-friday-pc-gaming/pcmcat1759172707743.c?id=pcmcat1759172707743",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat287600050003&id=pcat17071&qp=condition_facet%3DCondition%7ENew%5Ecurrentoffers_facet%3DCurrent+Deals%7EOn+Sale&st=categoryid%24pcmcat287600050003",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat304600050011&id=pcat17071&qp=currentoffers_facet=Current%20Deals%7EOn%20Sale&st=categoryid%24pcmcat304600050011"
]

HEADLESS = os.getenv('BESTBUY_HEADLESS', 'true').lower() == 'true'

def extract_json_data_from_html(html_content):
    """
    Best Buy embeds product data in JSON format within script tags.
    This extracts and parses that data.
    """
    products = []
    
    # NOTE: This function is a backup and not used in the main scraper since GraphQL parsing is more reliable.
    # Look for JSON data in script tags or window objects
    # Pattern 1: Look for productBySkuId data in the HTML itself, this was my workaround before GraphQL parsing (Still implemented for redundancy)
    pattern = r'"productBySkuId":\s*(\{[^}]*"skuId":"(\d+)"[^}]*\})'
    matches = re.finditer(pattern, html_content, re.DOTALL)
    
    seen_skus = set()
    
    for match in matches:
        sku_id = match.group(2)
        if not sku_id:
            print(f"  Warning: Empty SKU ID found, skipping")
            continue
        
        if sku_id in seen_skus:
            continue
        seen_skus.add(sku_id)
        
        # NOTE: Extracts the full product JSON block
        # NOTE: Find the complete JSON object for this product
        start_pos = match.start()
        # NOTE: Look backward to find the opening of the product object
        search_text = html_content[max(0, start_pos-5000):start_pos+10000]
    
        # NOTE: Tries to find complete product JSON (rudimentary method but so far its been working decently for most pages)
        product_pattern = rf'"skuId":"{sku_id}"[^{{}}]*(?:"name":\{{"__typename":"ProductName","short":"([^"]+)"\}})[^{{}}]*(?:"customerPrice":([0-9.]+))?'
        product_match = re.search(product_pattern, search_text)
        
        if not product_match:
            print(f"No product data found for SKU {sku_id}")
            continue
        
        title = product_match.group(1)
        price = product_match.group(2)
        
        if not title or not price:
            print(f"Missing title or price for SKU {sku_id}")
            continue
        
        # NOTE: Conversion method to conver price(s) to floats.
        # NOTE: Validates price can be converted to float, after having down so
        if not price.replace('.', '', 1).isdigit():
            print(f"Current price format needs to be adjusted '{price}' for SKU {sku_id}")
            continue
        
        products.append({
            'sku_id': sku_id,
            'title': title,
            'price': float(price),
            'link': f"https://www.bestbuy.com/site/-/{sku_id}.p"
        })
    
    return products

def extract_products_from_json_ld(html_content):
    """Extract products from JSON-LD structured data (another format Best Buy might use)"""
    products = []
    
    # NOTE: Looks for JSON-LD script tags
    json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(json_ld_pattern, html_content, re.DOTALL)
    
    for json_str in matches:
        if not json_str or not json_str.strip():
            print("  Warning: Empty JSON-LD content, skipping")
            continue
        
        # NOTE: Its a JSON structure cleanser, that is used before officially parsing, to avoid common pitfalls.
        json_str_clean = json_str.strip()
        if not json_str_clean.startswith(('{', '[')):
            print(f"  Warning: Invalid JSON-LD format (doesn't start with {{ or [)")
            continue
        
        data = json.loads(json_str_clean)
        
        # Check if it's a product list
        if not isinstance(data, dict):
            print(f"  Warning: JSON-LD is not a dict, got {type(data).__name__}")
            continue
        
        if data.get('@type') != 'ItemList':
            continue  # Not a product list, skip silently
        
        for item in data.get('itemListElement', []):
            product_data = item.get('item', {})
            if not product_data:
                continue
            
            # NOTE:validate check that ensures that data was populated in the required fields
            if not product_data.get('name') or not product_data.get('url'):
                print(f"  Warning: JSON-LD product missing name or URL")
                continue
            
            products.append({
                'title': product_data.get('name'),
                'price': product_data.get('offers', {}).get('price'),
                'link': product_data.get('url'),
                'sku_id': product_data.get('sku')
            })
    
    return products

def parse_graphql_responses(html_content):
    """
    Best Buy uses GraphQL. Extract product data from Apollo/GraphQL responses -- this was from my research and observations while trying to understand how to webscrape Best Buy by using what was
    embedded in the page HTML. This was my workaround to get reliable data, as otherwise traditional DOM Methods were not effective at all. 
    """
    products = []
    
    # NOTE: Pattern to match the GraphQL product data structure
    # Look for Product objects with skuId, then find the closest name and price in a window around it
    # Also extract regularPrice to calculate discounts as bestbuy is really good at advertising their sale prices without context.
    
    # NOTE: First, using regex, I wanted to find all Product blocks with skuId, as this was the best way to pair them together.
    sku_pattern = r'"__typename":"Product"[^{}]{0,200}?"skuId":"([^"]+)"'
    sku_matches = list(re.finditer(sku_pattern, html_content))
    
    print(f"Found {len(sku_matches)} potential products with SKU IDs")
    
    seen_skus = set()
    
    # NOTE: For each SKU, search for name and price in a window around it
    for sku_match in sku_matches:
        sku_id = sku_match.group(1)
        
        # NOTE: Validates SKU ID
        if not sku_id or sku_id in seen_skus:
            continue
        
        # Get a window of text around this SKU (3000 chars forward)
        start_pos = sku_match.start()
        search_window = html_content[start_pos:start_pos+3000]
        
        # NOTE: product name within this window (more flexible pattern)
        name_pattern = r'"name":\{"__typename":"ProductName","short":"([^"]+)"'
        name_match = re.search(name_pattern, search_window)
        
        # NOTE: customer price (current/sale price)
        price_pattern = r'"customerPrice":([0-9.]+)'
        price_match = re.search(price_pattern, search_window)
        
        # NOTE: Looks for regular price (original/non-sale price)
        regular_price_pattern = r'"regularPrice":([0-9.]+)'
        regular_price_match = re.search(regular_price_pattern, search_window)
        
        if not name_match or not price_match:
            continue
        
        title = name_match.group(1).replace('\u0026', '&')
        price_str = price_match.group(1)
        regular_price_str = regular_price_match.group(1) if regular_price_match else None
        
        # NOTE: Validates the extracted data, at least by pairing it, its otherwise not super useful but a simple quality of life check / assessment. 
        if not title or not price_str:
            continue
        
        seen_skus.add(sku_id)
        
        # NOTE: Cleans the title (ie, however, some titles are still quite short, so I filter those out later)
        title = title.strip()
        title = re.sub(r'\\u[\da-fA-F]{4}', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        if len(title) < 10:
            continue
        
        # NOTE: Failsafe to ensure that price format isn't corrupted on return
        if not price_str.replace('.', '', 1).isdigit():
            continue
        
        current_price = float(price_str)
        original_price = None
        discount_percent = None
        
        # NOTE: Calculates discount if regular price exists and is higher than current price, sort of works on the targeted cards, but not always perfect.
        if regular_price_str and regular_price_str.replace('.', '', 1).isdigit():
            regular_price_val = float(regular_price_str)
            if regular_price_val > current_price:
                original_price = regular_price_val
                discount_percent = round(((original_price - current_price) / original_price) * 100, 1)
                
                # Sanity check: reject if discount > 95% (likely data error)
                if discount_percent > 95:
                    print(f"  Warning: Suspicious {discount_percent:.1f}% discount on '{title[:50]}...' - skipping")
                    continue
        
        products.append({
            'sku_id': sku_id,
            'title': title,
            'price': current_price,
            'price_text': f"${price_str}",
            'link': f"https://www.bestbuy.com/site/-/{sku_id}.p?skuId={sku_id}",
            'original_price': original_price,
            'discount_percent': discount_percent
        })
    
    print(f"Extracted {len(products)} products from GraphQL data")
    if products:
        discounted_count = sum(1 for p in products if p.get('discount_percent'))
        print(f"  └─ {discounted_count} products have discount data")

    return products

@browser(
    headless=HEADLESS, 
    reuse_driver=True,
    block_images=False,
    wait_for_complete_page_load=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

# NOTE: this is my primary scraping function that uses the above extraction methods, if it weren't for GraphQL, I would have to rely on DOM parsing which is less reliable for Best Buy from my experience. 

def scrape_bestbuy_deals_api(driver: Driver, url: str):
    """
    This is my most reliable rendition of webscraping bestbuy, no other method has come closer than this one. 
    Scrape Best Buy by extracting JSON/GraphQL data from the page source.
    This is more reliable than DOM parsing since Best Buy uses React/GraphQL.
    """
    print(f"\nLoading URL: {url}")
    driver.get(url)
    
    # NOTE: Waits for the page to load, 3-5 seconds has been somewhat okay for my usage cases, however, may need adjustment based on network conditions. (I've had tons of issues, so anything more than 5 seconds is actually preferable)
    print("Waiting for page to load...")
    time.sleep(5)
    
    # NOTE: Scrolls a bit to trigger any lazy content (site loads super duper slow otherwise)
    # NOTE: Checks if driver(s) supports scrolling before attempting to do so ..., (some drivers have not accepted this command at times on a per page basis.)
    if hasattr(driver, 'scroll_to_bottom') and callable(driver.scroll_to_bottom):
        driver.scroll_to_bottom()
        time.sleep(2)
    else:
        print("Scroll to Bottom is not supported on this webpage.")
    
    html_content = driver.page_html
    print(f"Page loaded ({len(html_content)} potential characters)")
    
    # NOTE: My script tries multiple extraction methods, however, this isn't strictly necessary since GraphQL is usually sufficient (two methods for redundancy)
    products = []
    
    # NOTE: Method 1: GraphQL responses (most reliable for Best Buy in my experience)
    print("  Extracting GraphQL data...")
    graphql_products = parse_graphql_responses(html_content)
    products.extend(graphql_products)
    
    # NOTE: Method 2: JSON-LD structured data (backup method as some pages do not have GraphQL data)   
    if len(products) == 0:
        print("  Trying JSON-LD extraction...")
        jsonld_products = extract_products_from_json_ld(html_content)
        products.extend(jsonld_products)
    
    # NOTE: If still no products, save HTML for debugging (error logging with a saved file)
    # NOTE: In real usage, I acknowledge this may not be ideal for production due to storage concerns
    if len(products) == 0:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        err_dir = Path(__file__).resolve().parent / "error_logs" / ts
        
        # NOTE: Checks if parent directory is writable, if not then it skips the whole saving state for this session (HMTL WISE that is)
        parent_dir = Path(__file__).resolve().parent
        if not os.access(str(parent_dir), os.W_OK):
            print(f"Cannot write to {parent_dir}, HTML will not be saved in this session.")
            return []
        
        err_dir.mkdir(parents=True, exist_ok=True)
        html_file = err_dir / "page.html"
        
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"No products extracted. Saved HTML to {html_file}")
        return []
    
    # NOTE: Updated: Detects category from URL & assign category label
    # NOTE: My pipeline sort of works right now, however, adverk and other content has in some cases ruined or caused mislabeling issues.
    category = 'Tech Deals'
    url_l = url.lower()
    if 'gaming' in url_l or 'pcmcat1591132221892' in url_l:
        category = 'Gaming'
    elif 'laptop' in url_l:
        category = 'Laptops'
    elif 'monitor' in url_l:
        category = 'Monitors'
    elif 'tv' in url_l:
        category = 'TVs'
    
    # NOTE: Convert to deal format, stored in a list, filtering out incomplete entries.
    deals = []


    for product in products:
        if not product.get('title') or not product.get('price'):
            continue
        
        deals.append({
            'title': product['title'],
            'link': product['link'],
            'price': product.get('price_text', f"${product['price']}"),
            'price_numeric': product['price'],
            'price_text': product.get('price_text', f"${product['price']}"),
            'category': category,
            'website': 'bestbuy',
            'scraped_at': datetime.now().isoformat(),
            'original_price': product.get('original_price'),
            'discount_percent': product.get('discount_percent'),
            'rating': None,
            'reviews_count': None,
        })
    
    print(f"Extracted {len(deals)} valid deals")
    return deals

def main():
    print("Best Buy API Scraper (GraphQL/JSON Extraction)")
    print("=" * 60)
    print("This scraper extracts product data from Best Buy's embedded")
    print("GraphQL API responses instead of parsing the DOM.")
    print("=" * 60)
    
    all_deals = []
    successful_urls = 0
    failed_urls = []
    
    for url in BESTBUY_URLS:
        # NOTE: Validates URL format before processing
        if not url or not url.startswith('http'):
            print(f"Invalid URL format: {url[:80]}")
            failed_urls.append(url[:80])
            continue
        
        deals = scrape_bestbuy_deals_api(url)
        
        if deals:
            all_deals.extend(deals)
            successful_urls += 1
            print(f"Successful scraping session: {len(deals)} deals from URL #{successful_urls}")
        else:
            print(f"Failed session: No deals from this URL")
            failed_urls.append(url[:80])
    
    # NOTE: Created a summary of the scraping session with vitals that I could reason through that are impactful to quicking assessing data quality.

    print(f"Summary: {successful_urls}/{len(BESTBUY_URLS)} URLs successful")
    print(f"Total deals extracted: {len(all_deals)}")
    print(f"Deals with prices: {len([d for d in all_deals if d.get('price_numeric')])}")
    
    if failed_urls:
        print(f"Failed URLs: {len(failed_urls)}")
    
    if not all_deals:
        print("\nNo deals were collected")
        return
    
    # NOTE: Saves my output from the webscraper via a pipeline
    project_root = Path(__file__).resolve().parent
    pipeline = DealsDataPipeline(output_dir=str(project_root / "output"))
    result = pipeline.process_deals(all_deals, csv_prefix="bestbuy_api")
    
    print(f"\nCSV saved to: {result['csv']}")
    print(f"Database entries added: {result['database_rows_added']}")

if __name__ == "__main__":
    main()
