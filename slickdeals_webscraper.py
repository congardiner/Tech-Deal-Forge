from pathlib import Path

from botasaurus.browser import browser, Driver
from botasaurus.soupify import soupify
from data_pipeline import DealsDataPipeline
import argparse
import re
from datetime import datetime


# Slickdeals categories to crawl (frontpage + specific categories of interest that I've isolated)

CATEGORY_URLS = [
    "https://slickdeals.net/computer-deals/",
    "https://slickdeals.net/deals/tech/?filters%5Brating%5D%5B%5D=frontpage",
    "https://slickdeals.net/laptop-deals/?filters%5Brating%5D%5B%5D=frontpage",
    "https://slickdeals.net/deals/computer-parts/?filters%5Brating%5D%5B%5D=frontpage",
    "https://slickdeals.net/deals/video-card/?filters%5Brating%5D%5B%5D=frontpage",
    "https://slickdeals.net/deals/processor/?filters%5Brating%5D%5B%5D=frontpage",
    "https://slickdeals.net/deals/memory/?filters%5Brating%5D%5B%5D=popular",
    "https://slickdeals.net/monitor-deals/",
    "https://slickdeals.net/deals/software/?filters%5Brating%5D%5B%5D=frontpage&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/motherboard/",
    "https://slickdeals.net/deals/power-supply/",
    "https://slickdeals.net/computer-case-deals/?filters%5Brating%5D%5B%5D=frontpage&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/drives/",
    "https://slickdeals.net/deals/desktop/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/servers/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/printer/",
    "https://slickdeals.net/monitor-deals/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/laptop-deals/",
    "https://slickdeals.net/deals/phone/?filters%5Brating%5D%5B%5D=frontpage&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/unlocked-phones/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/smart-watch/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/audio/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/tv-deals/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/cell-phone-deals/",
    "https://slickdeals.net/computer-deals/?filters%5Brating%5D%5B%5D=frontpage&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/cellphone-providers/?filters%5Brating%5D%5B%5D=frontpage&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/deals/education/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=",
    "https://slickdeals.net/ps4-deals/",
    "https://slickdeals.net/xbox-one-deals/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # Xbox Deals
    "https://slickdeals.net/deals/ps5-deals/?filters%5Brating%5D%5B%5D=5&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # PS5 deals
    "https://slickdeals.net/deals/nintendo-switch/?filters%5Brating%5D%5B%5D=5&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # Nintendo Switch
    "https://slickdeals.net/deals/console/?filters%5Brating%5D%5B%5D=5&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # Video Game Consoles
    "https://slickdeals.net/deals/apple/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # Popular Apple Product Deals
    "https://slickdeals.net/deals/computer-parts/?filters%5Brating%5D%5B%5D=5&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # Computer Parts Deals
    "https://slickdeals.net/deals/smartphone/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D=", # More Phone Deals
    "https://slickdeals.net/deals/virtual-reality/",
    "https://slickdeals.net/tv-deals/?filters%5Brating%5D%5B%5D=popular&filters%5Bprice%5D%5Bmin%5D=&filters%5Bprice%5D%5Bmax%5D="
]


EXCLUDED_DOMAINS = [
    "adzerk.net",
    "doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
    "amazon-adsystem.com",
]


def should_exclude_link(url: str) -> bool:
    if not url:
        return True
    for domain in EXCLUDED_DOMAINS:
        if domain in url:
            return True
    if url.startswith("javascript:") or url.startswith("mailto:"):
        return True
    return False


def extract_price_details(price_text: str | None) -> dict[str, float | str | None]:
    if not price_text:
        return {
            "price_text": None,
            "price_numeric": None,
            "original_price": None,
            "discount_percent": None,
        }

    clean_text = price_text.strip()
    prices = re.findall(r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", clean_text.replace(",", ""))

    price_numeric = float(prices[0]) if prices else None
    original_price = float(prices[1]) if len(prices) > 1 else None
    discount_percent = None

    if price_numeric and original_price and original_price > price_numeric:
        discount_percent = round(((original_price - price_numeric) / original_price) * 100, 1)

    if not discount_percent:
        match = re.search(r"(\d+)%\s*off", clean_text, re.IGNORECASE)
        if match:
            discount_percent = float(match.group(1))

    return {
        "price_text": clean_text,
        "price_numeric": price_numeric,
        "original_price": original_price,
        "discount_percent": discount_percent,
    }


def extract_rating_info(card) -> tuple[float | None, int | None]:
    rating = None
    reviews = None

    rating_elem = card.select_one('[data-test="bp-c-card-reviewRating"]') or card.select_one(
        ".bp-c-rating"
    )
    if rating_elem:
        match = re.search(r"(\d+\.?\d*)", rating_elem.get_text(strip=True))
        if match:
            rating = float(match.group(1))

    reviews_elem = card.select_one('[data-test="bp-c-card-reviewCount"]')
    if reviews_elem:
        match = re.search(r"(\d+)", reviews_elem.get_text(strip=True))
        if match:
            reviews = int(match.group(1))

    return rating, reviews



def extract_image_url(card) -> str | None:
    for selector in [
        "img[data-src]",
        "img[data-original]",
        "img[src]",
    ]:
        image = card.select_one(selector)
        if not image:
            continue
        src = image.get("data-src") or image.get("data-original") or image.get("src")
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = f"https://slickdeals.net{src}"
        return src
    return None


def extract_description(card) -> str | None:
    desc_elem = card.select_one('[data-test="bp-c-card-description"]') or card.select_one(
        ".bp-c-card_description"
    )
    if not desc_elem:
        return None

    text = re.sub(r"\s+", " ", desc_elem.get_text(strip=True))
    return text[:250] if text else None


def get_category_from_url(url: str) -> str:
    slug = url.split("//", 1)[-1]
    slug = slug.split("?", 1)[0]
    slug = slug.strip("/")
    parts = slug.split("/")
    if len(parts) > 1:
        return parts[-1].replace("-deals", "").replace("deal", "deal").replace("deal", "deal")
    return slug


@browser(headless=True, reuse_driver=True)
def scrape_tech_deals_with_pipeline(driver: Driver, data=None):
    all_deals: list[dict] = []
    excluded_count = 0

    for url in CATEGORY_URLS:
        print(f"Scraping: {url}")
        driver.get(url)
        driver.sleep(3)

        soup = soupify(driver.page_html)
        cards = soup.select(".bp-c-card")
        print(f"Found {len(cards)} cards on {url}")

        for card in cards:
            title_link = card.select_one('a[href*="/f/"]') or card.find("a", href=True)
            if not title_link:
                continue

            href = title_link.get("href", "")
            full_link = href if href.startswith("http") else f"https://slickdeals.net{href}"
            if should_exclude_link(full_link):
                excluded_count += 1
                continue

            # Build best-effort title
            title_text = (
                title_link.get("title")
                or title_link.get("aria-label")
                or title_link.get_text(strip=True)
                or ""
            ).strip()

            if not title_text and href.startswith("/f/"):
                slug = href.split("/f/", 1)[-1]
                title_text = slug.replace("-", " ").title()[:100]

            if len(title_text) < 5:
                continue

            price_elem = card.select_one(".bp-p-dealCard_price, .bp-c-card_price")
            price_text = price_elem.get_text(strip=True) if price_elem else None
            price_details = extract_price_details(price_text)

            rating, reviews_count = extract_rating_info(card)
            image_url = extract_image_url(card)
            description = extract_description(card)
            category = get_category_from_url(url)

            deal = {
                "title": title_text,
                "price": price_details["price_text"],
                "price_text": price_details["price_text"],
                "price_numeric": price_details["price_numeric"],
                "link": full_link,
                "category": category,
                "website": "slickdeals",
                "image_url": image_url,
                "description": description,
                "discount_percent": price_details["discount_percent"],
                "original_price": price_details["original_price"],
                "rating": rating,
                "reviews_count": reviews_count,
                "scraped_at": datetime.now().isoformat(),
            }
            all_deals.append(deal)

    print(f"Total deals scraped: {len(all_deals)} (excluded {excluded_count} promotional links)")
    return all_deals


def main():
    parser = argparse.ArgumentParser(description="Scrape Slickdeals tech feeds and push through the pipeline")
    parser.add_argument("--min-price", type=float, help="Minimum price filter")
    parser.add_argument("--max-price", type=float, help="Maximum price filter")
    parser.add_argument("--keywords", nargs="+", help="Keywords to include")
    parser.add_argument("--exclude", nargs="+", help="Keywords to exclude")
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "database", "all"],
        default="all",
        help="Output format",
    )
    parser.add_argument("--no-scrape", action="store_true", help="Skip scraping, use existing data")

    args = parser.parse_args()

    # Initialize pipeline targeting the project's output folder explicitly
    # This ensures the same output/deals.db file is used regardless of CWD
    project_root = Path(__file__).resolve().parent
    pipeline = DealsDataPipeline(output_dir=str(project_root / "output"), use_mysql=False)

    if args.no_scrape:
        # If skipping scraping, try to proceed without loading a JSON cache.
        # We'll just exit early since there's no fresh data to process.
        print("--no-scrape specified and no cached CSV loader implemented. Exiting.")
        return
    else:
        deals = scrape_tech_deals_with_pipeline()

    if not deals:
        print("No deals found")
        return

    # No JSON dump: we now rely on the pipeline to write CSV and DB directly

    filter_args = {}
    if args.min_price is not None:
        filter_args["min_price"] = args.min_price
    if args.max_price is not None:
        filter_args["max_price"] = args.max_price
    if args.keywords:
        filter_args["keywords"] = args.keywords
    if args.exclude:
        filter_args["exclude_keywords"] = args.exclude

    print(f"Processing {len(deals)} deals")
    if filter_args:
        print(f"Filters: {filter_args}")

    results = pipeline.process_deals(deals, csv_prefix="slickdeals", **filter_args)

    print("-" * 60)
    print("Processing Results")
    summary = results["summary"]
    print(f"Total deals: {summary['total_deals']}")
    print(f"Deals with prices: {summary['deals_with_prices']}")

    if summary["price_range"]["min"] is not None:
        print(f"Price range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
        print(f"Average price: ${summary['price_range']['avg']:.2f}")

    print(f"Categories: {summary['categories']}")

    if "csv" in results:
        print(f"CSV saved to: {results['csv']}")
    if "parquet" in results:
        print(f"Parquet: {results['parquet']}")
    if "database_rows_added" in results:
        print(f"Database: {results['database_rows_added']} rows added")

    print("\nSample Deals:")
    clean_df = pipeline.clean_data(deals)
    if filter_args:
        clean_df = pipeline.filter_deals(clean_df, **filter_args)

    for idx, (_, deal) in enumerate(clean_df.head(5).iterrows(), start=1):
        price_str = f"${deal['price_numeric']:.2f}" if deal.get("price_numeric") else "No price"
        print(f"{idx}. {deal['title'][:80]} - {price_str}")


if __name__ == "__main__":
    main()