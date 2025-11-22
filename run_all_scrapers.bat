@echo off
echo ============================================
echo DealForge - Run All Scrapers
echo ============================================
echo.

echo [1/3] Scraping SlickDeals...
python slickdeals_webscraper.py --format database
echo.

echo [2/3] Scraping Best Buy...
python bestbuy_api_scraper.py
echo.

echo [3/3] Preparing ML Data...
python export_deals_for_ml.py
echo.


echo ============================================
echo Done! All data saved to:
echo   - CSV: output\*.csv
echo   - Database: output\deals.db
echo   - ML Data: output\ml_data\*.csv
echo ============================================

REM Log completion with timestamp
echo [%date% %time%] Scraping completed >> scraper_log.txt