import os
import re
import json
import time
from datetime import datetime
from botasaurus.browser import browser, Driver
from data_pipeline import DealsDataPipeline
from pathlib import Path


# Best Buy Deal URLs (ie, tested and validated over the span of my project)
BESTBUY_URLS = [
    # "https://www.bestbuy.com/site/promo/black-friday-laptop-computer-deals-1",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1591132221892&id=pcat17071&qp=currentoffers_facet=Current+Deals%7EOn+Sale&st=pcmcat1591132221892_categoryid%24abcat0500000",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1591132221892&id=pcat17071&qp=currentoffers_facet%3DCurrent+Deals%7EBlack+Friday+Deal&st=pcmcat1591132221892_categoryid%24abcat0500000",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat1720706651516&id=pcat17071&qp=currentoffers_facet%3DSeasonal+Savings%7EBlack+Friday+Deal&st=pcmcat1720706651516_categoryid%24pcmcat156400050037",
    "https://www.bestbuy.com/site/pc-gaming/gaming-laptops/pcmcat287600050003.c?id=pcmcat287600050003",
    "https://www.bestbuy.com/site/computers-pcs/computer-monitors/abcat0509000.c?id=abcat0509000",
    "https://www.bestbuy.com/site/searchpage.jsp?browsedCategory=pcmcat138500050001&id=pcat17071&qp=currentoffers_facet%3DCurrent+Deals%7EOn+Sale%5Econdition_facet%3DCondition%7ENew&st=categoryid%24pcmcat138500050001",
    "https://www.bestbuy.com/site/xbox-series-x-and-s/xbox-series-x-and-s-consoles/pcmcat1586900952752.c?id=pcmcat1586900952752",
    "https://www.bestbuy.com/site/all-electronics-on-sale/all-wearable-technology-on-sale/pcmcat1690897435393.c?id=pcmcat1690897435393"
]

HEADLESS = os.getenv('BESTBUY_HEADLESS', 'true').lower() == 'true'

def extract_json_data_from_html(html_content):
    """
    Best Buy embeds product data in JSON format within script tags.
    This extracts and parses that data.
    """
    products = []
    
    # Look for JSON data in script tags or window objects
    # Pattern 1: Look for productBySkuId data
    pattern = r'"productBySkuId":\s*(\{[^}]*"skuId":"(\d+)"[^}]*\})'
    matches = re.finditer(pattern, html_content, re.DOTALL)
    
    seen_skus = set()
    
    for match in matches:
        try:
            sku_id = match.group(2)
            if sku_id in seen_skus:
                continue
            seen_skus.add(sku_id)
            
            # Extract the full product JSON block
            # Find the complete JSON object for this product
            start_pos = match.start()
            # Look backward to find the opening of the product object
            search_text = html_content[max(0, start_pos-5000):start_pos+10000]
            
            # Try to find complete product JSON
            product_pattern = rf'"skuId":"{sku_id}"[^{{}}]*(?:"name":\{{"__typename":"ProductName","short":"([^"]+)"\}})[^{{}}]*(?:"customerPrice":([0-9.]+))?'
            product_match = re.search(product_pattern, search_text)
            
            if product_match:
                title = product_match.group(1)
                price = product_match.group(2)
                
                if title and price:
                    products.append({
                        'sku_id': sku_id,
                        'title': title,
                        'price': float(price),
                        'link': f"https://www.bestbuy.com/site/-/{sku_id}.p"
                    })
        except Exception as e:
            continue
    
    return products

def extract_products_from_json_ld(html_content):
    """Extract products from JSON-LD structured data (another format Best Buy might use)"""
    products = []
    
    # Look for JSON-LD script tags
    json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(json_ld_pattern, html_content, re.DOTALL)
    
    for json_str in matches:
        try:
            data = json.loads(json_str)
            # Check if it's a product list
            if isinstance(data, dict) and data.get('@type') == 'ItemList':
                for item in data.get('itemListElement', []):
                    product_data = item.get('item', {})
                    if product_data:
                        products.append({
                            'title': product_data.get('name'),
                            'price': product_data.get('offers', {}).get('price'),
                            'link': product_data.get('url'),
                            'sku_id': product_data.get('sku')
                        })
        except:
            continue
    
    return products

def parse_graphql_responses(html_content):
    """
    Best Buy uses GraphQL. Extract product data from Apollo/GraphQL responses
    embedded in the page HTML.
    """
    products = []
    
    # Pattern to match the GraphQL product data structure we saw in the error log
    # Looking for: "Product","skuId":"XXXXX"..."name":{"__typename":"ProductName","short":"PRODUCT NAME"}..."customerPrice":999.99
    
    product_pattern = r'"__typename":"Product"[^}]*?"skuId":"([^"]+)".*?"name":\{"__typename":"ProductName","short":"([^"]+)"\}.*?"customerPrice":([0-9.]+)'
    
    matches = re.finditer(product_pattern, html_content, re.DOTALL)
    
    seen_skus = set()
    
    for match in matches:
        try:
            sku_id = match.group(1)
            title = match.group(2).replace('\u0026', '&')  # Fix HTML entities
            price_str = match.group(3)
            
            if sku_id in seen_skus or not title or not price_str:
                continue
            
            seen_skus.add(sku_id)
            
            # Clean the title
            title = title.strip()
            # Remove HTML artifacts
            title = re.sub(r'\\u[\da-fA-F]{4}', ' ', title)
            title = re.sub(r'\s+', ' ', title).strip()
            
            if len(title) < 10:  # Skip if title too short
                continue
            
            products.append({
                'sku_id': sku_id,
                'title': title,
                'price': float(price_str),
                'price_text': f"${price_str}",
                'link': f"https://www.bestbuy.com/site/-/{sku_id}.p?skuId={sku_id}"
            })
            
        except Exception as e:
            continue
    
    print(f"   Extracted {len(products)} products from GraphQL data")
    return products

@browser(
    headless=HEADLESS, 
    reuse_driver=True,
    block_images=False,
    wait_for_complete_page_load=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
def scrape_bestbuy_deals_api(driver: Driver, url: str):
    """
    Scrape Best Buy by extracting JSON/GraphQL data from the page source.
    This is more reliable than DOM parsing since Best Buy uses React/GraphQL.
    """
    print(f"\nLoading URL: {url}")
    driver.get(url)
    
    # Wait for page to load
    print("  Waiting for page to load...")
    time.sleep(5)
    
    # Scroll a bit to trigger any lazy content
    try:
        driver.scroll_to_bottom()
        time.sleep(2)
    except:
        pass
    
    html_content = driver.page_html
    print(f"  Page loaded ({len(html_content)} chars)")
    
    # Try multiple extraction methods
    products = []
    
    # Method 1: GraphQL responses (most reliable for Best Buy)
    print("  Extracting GraphQL data...")
    graphql_products = parse_graphql_responses(html_content)
    products.extend(graphql_products)
    
    # Method 2: JSON-LD structured data (backup)
    if len(products) == 0:
        print("  Trying JSON-LD extraction...")
        jsonld_products = extract_products_from_json_ld(html_content)
        products.extend(jsonld_products)
    
    # If still no products, save HTML for debugging
    if len(products) == 0:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        err_dir = Path(__file__).resolve().parent / "error_logs" / ts
        try:
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "page.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"  ⚠️ No products extracted. Saved HTML to {err_dir / 'page.html'}")
        except:
            pass
        return []
    
    # Detect category from URL
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
    
    # Convert to deal format
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
            'original_price': None,
            'discount_percent': None,
            'rating': None,
            'reviews_count': None,
        })
    
    print(f"  ✓ Extracted {len(deals)} valid deals")
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
        try:
            deals = scrape_bestbuy_deals_api(url)
            if deals:
                all_deals.extend(deals)
                successful_urls += 1
                print(f"✓ Success: {len(deals)} deals from URL #{successful_urls}")
            else:
                print(f"✗ Failed: No deals from this URL")
                failed_urls.append(url[:80])
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            failed_urls.append(url[:80])
    
    print("\n" + "=" * 60)
    print(f"Summary: {successful_urls}/{len(BESTBUY_URLS)} URLs successful")
    print(f"Total deals extracted: {len(all_deals)}")
    print(f"Deals with prices: {len([d for d in all_deals if d.get('price_numeric')])}")
    
    if failed_urls:
        print(f"Failed URLs: {len(failed_urls)}")
    
    if not all_deals:
        print("\n⚠️ Warning: No deals collected")
        print("Try running with BESTBUY_HEADLESS=false to see what's happening")
        return
    
    # Save via pipeline
    project_root = Path(__file__).resolve().parent
    pipeline = DealsDataPipeline(output_dir=str(project_root / "output"), use_mysql=False)
    result = pipeline.process_deals(all_deals, csv_prefix="bestbuy_api")
    
    print(f"\n📁 CSV saved to: {result['csv']}")
    print(f"💾 Database entries added: {result['database_rows_added']}")
    print("\n✅ Scraping complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
