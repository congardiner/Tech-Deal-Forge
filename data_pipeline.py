import pandas as pd
import sqlite3
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging


# NOTE: REGEX was used extensively as this was a huge part of the data cleaning and validation process.
# NOTE: I've created a ton of different edge cases to ensure that the data is clean and standardized for production use, especially with price extraction and validation, as this was a repeat issue that I kept running into.
# NOTE: In addition to this, I've tried my best to ensure that all rows are properly timestamped for historical tracking, which is a key requirement for this project, as I wanted to build and track price trends over time...
# NOTE: Data pipeline class for processing and storing deal data, with historical tracking and filtering capabilities, with comments enclosed where applicable for clarity.



class DealsDataPipeline:
    """
    Stores data in SQLite with support for historical tracking and filtering, which I've used extensively in my data analysis and reporting, however, this is not a fool-proof system, as I've had constant revisions. Just something to keep in mind. 
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # NOTE: Setup logging for the sqlite database operations
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # NOTE: SQLite database configuration (now initialized in my class constructor)
        self.db_path = self.output_dir / "deals.db"
        self.logger.info(f"Using SQLite database: {self.db_path}")
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema for deal tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create deals table with indexes for performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price TEXT,
                link TEXT NOT NULL,
                category TEXT,
                price_text TEXT,
                price_numeric REAL,
                website TEXT DEFAULT 'slickdeals',
                image_url TEXT,
                description TEXT,
                discount_percent REAL,
                original_price REAL,
                rating REAL,
                reviews_count INTEGER,
                availability TEXT,
                in_stock BOOLEAN DEFAULT 1,
                is_active BOOLEAN DEFAULT 1,
                scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # NOTE: Creates indexes for common query patterns (OLD logic retained for clarity, as I've revised this multiple times)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON deals(price_numeric)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON deals(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_website ON deals(website)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scraped_at ON deals(scraped_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_link ON deals(link)")
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"SQLite database initialized: {self.db_path}")
    
    def clean_data(self, deals: List[Dict]) -> pd.DataFrame:
        """Clean and standardize deals data for production use"""
        df = pd.DataFrame(deals)
        
        if df.empty:
            self.logger.warning("No deals data to clean")
            return df
        
        # NOTE: Duplicates are intentionally KEPT for historical price tracking
        # Each scrape creates a new entry with timestamp for timeline analysis
        
        # NOTE: Cleans and standardize titles as much as possible using some super basic regex statements
        df['title'] = df['title'].str.strip()
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
        
        # NOTE: Handles price columns - support both old 'price' and new 'price_text'/'price_numeric' format
        if 'price' in df.columns and 'price_text' not in df.columns:
            df['price_numeric'] = df['price'].apply(self._extract_numeric_price)
            df['price_text'] = df['price']
        elif 'price_text' in df.columns and 'price_numeric' not in df.columns:
            df['price_numeric'] = df['price_text'].apply(self._extract_numeric_price)
        

        # NOTE: Cleans the created using some simple renaming via regex statements again against categories
        if 'category' in df.columns:
            df['category'] = df['category'].str.replace('https://slickdeals.net/', '', regex=False)
            df['category'] = df['category'].str.replace('/', '', regex=False)
            df['category'] = df['category'].str.replace('?', '', regex=False)
            df['category'] = df['category'].str.strip()
        
        # NOTE: I made a way to add a scraping timestamp if not present
        if 'scraped_at' not in df.columns:
            df['scraped_at'] = datetime.now().isoformat()
        
        # NOTE: Filters out deals with very short titles (likely errors)
        original_count = len(df)
        df = df[df['title'].str.len() > 10]
        removed = original_count - len(df)
        
        if removed > 0:
            self.logger.info(f"Removed {removed} deals with invalid titles (too short)")
        
        self.logger.info(f"Cleaned data: {len(df)} deals ready for processing")
        return df
    
    def _extract_numeric_price(self, price_text: str) -> Optional[float]:
        """Extract numeric price from price text with validation"""
        if not price_text or pd.isna(price_text):
            return None
        
        # NOTE: Removes currency symbols and extracts numbers using regex, no issues to report in my edge cases.
        price_clean = re.sub(r'[^\d.,]', '', str(price_text))
        
        if not price_clean:
            return None
        
        # NOTE: Handles comma-separated thousands, ensuring proper decimal handling, as this was a primary issue I kept running into.
        if ',' in price_clean and '.' in price_clean:
            # Format: 1,299.99
            price_clean = price_clean.replace(',', '')
        elif ',' in price_clean and '.' not in price_clean:
            # Format: 1,299 (no decimal)
            price_clean = price_clean.replace(',', '')
        
        # NOTE: REGEX CONTINUES: Validate the cleaned price is a valid number
        if not price_clean.replace('.', '', 1).isdigit():
            return None
        
        price_value = float(price_clean)
        
        # NOTE: Sanity check: reject unrealistic prices, there were a few ads that for some reason were initially scraped with prices like $9999999.99)
        if price_value <= 0 or price_value > 1000000:
            return None
        
        return price_value
    
    def filter_deals(self, df: pd.DataFrame, **filters) -> pd.DataFrame:
        """Filter deals based on various criteria"""
        filtered_df = df.copy()
        
        # Price range filter
        if 'min_price' in filters and filters['min_price'] is not None:
            filtered_df = filtered_df[
                (filtered_df['price_numeric'] >= filters['min_price']) | 
                (filtered_df['price_numeric'].isna())
            ]
        
        if 'max_price' in filters and filters['max_price'] is not None:
            filtered_df = filtered_df[
                (filtered_df['price_numeric'] <= filters['max_price']) | 
                (filtered_df['price_numeric'].isna())
            ]
        
        # NOTE: Category filter (no issues to report at this time of writing)
        if 'categories' in filters and filters['categories']:
            if isinstance(filters['categories'], str):
                filters['categories'] = [filters['categories']]
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # NOTE: Keyword filter in title, supports multiple keywords, 
        if 'keywords' in filters and filters['keywords']:
            keywords = filters['keywords']
            if isinstance(keywords, str):
                keywords = [keywords]
            
            # NOTE: Creates the same triangulated regex pattern for any keyword, making sure to escape special characters, although my edge cases should be minimal here.
            pattern = '|'.join([re.escape(keyword) for keyword in keywords])
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(pattern, case=False, na=False)
            ]
        
        # NOTE: Exclude keywords as specified have been excluded. 
        if 'exclude_keywords' in filters and filters['exclude_keywords']:
            exclude_keywords = filters['exclude_keywords']
            if isinstance(exclude_keywords, str):
                exclude_keywords = [exclude_keywords]
            
            pattern = '|'.join([re.escape(keyword) for keyword in exclude_keywords])
            filtered_df = filtered_df[
                ~filtered_df['title'].str.contains(pattern, case=False, na=False)
            ]
        
        self.logger.info(f"Filtered from {len(df)} to {len(filtered_df)} deals")
        return filtered_df
    
    def to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """Export deals to CSV file with timestamp"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deals_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Exported {len(df)} deals to CSV: {csv_path}")
        
        return csv_path
    
    def to_database(self, df: pd.DataFrame) -> int:
        """Save deals to SQLite database with historical tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rows_added = 0
        
        for _, row in df.iterrows():

            # NOTE: Inserts all deals for historical tracking (no duplicate prevention as each entry is timestamped for ensuring that there is a consistent log)
            cursor.execute("""
                INSERT INTO deals 
                (title, price_text, price_numeric, link, category, website, 
                 image_url, description, discount_percent, original_price, 
                 rating, reviews_count, availability, in_stock, scraped_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['title'], 
                row.get('price_text') or row.get('price'), 
                row.get('price_numeric'),
                row['link'], 
                row.get('category'),
                row.get('website', 'slickdeals'),
                row.get('image_url'),
                row.get('description'),
                row.get('discount_percent'),
                row.get('original_price'),
                row.get('rating'),
                row.get('reviews_count'),
                row.get('availability'),
                row.get('in_stock', True),
                row['scraped_at'],
                row.get('is_active', True)
            ))
            
            if cursor.rowcount > 0:
                rows_added += 1
        
        conn.commit()
        
        # NOTE: Logs total database size *after* insertions have already taken place
        cursor.execute("SELECT COUNT(*) FROM deals")
        total_rows = cursor.fetchone()[0]
        
        conn.close()
        
        self.logger.info(f"Added {rows_added} deals to database (Total: {total_rows})")
        return rows_added
            

    def get_deals_from_db(self, **filters) -> pd.DataFrame:
        """Retrieve deals from SQLite database with optional filters"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM deals WHERE is_active = 1"
        params = []
        
        if 'min_price' in filters and filters['min_price'] is not None:
            query += " AND (price_numeric >= ? OR price_numeric IS NULL)"
            params.append(filters['min_price'])
        
        if 'max_price' in filters and filters['max_price'] is not None:
            query += " AND (price_numeric <= ? OR price_numeric IS NULL)"
            params.append(filters['max_price'])
        
        if 'category' in filters and filters['category']:
            query += " AND category = ?"
            params.append(filters['category'])
        
        if 'website' in filters and filters['website']:
            query += " AND website = ?"
            params.append(filters['website'])
        
        query += " ORDER BY scraped_at DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        self.logger.info(f"Retrieved {len(df)} deals from database")
        return df
    
    def process_deals(self, deals_data: List[Dict], csv_prefix: Optional[str] = None, **filters) -> Dict:
        """
        Complete production pipeline: clean, filter, and export deals.
        
        Args:
            deals_data: List of deal dictionaries from scraper
            csv_prefix: Optional prefix for CSV filename (e.g., 'bestbuy_api')
            **filters: Optional filters to apply before export
        
        Returns:
            Dictionary with export results and summary statistics
        """

        df = self.clean_data(deals_data)
        
        if df.empty:
            self.logger.warning("No deals to process")
            return {
                'csv': None,
                'database_rows_added': 0,
                'summary': {'total_deals': 0}
            }
        
        # NOTE: Apply filters if provided (before export of the data)
        if filters:
            df = self.filter_deals(df, **filters)
        
        # NOTE: Export to all formats, which as of right now is just csv and to my mysql-lite database.
        results = {}
        
        # NOTE: CSV export with optional prefix applied to the filename, its a simple way to know where when the file itself was populated and from which source.
        # NOTE: It also helps with versioning and historical tracking of the data as well, as ML models and data analysis can be done on specific datasets after running my export from the export_deals_for_ml.py file.
        # NOTE: Timestamp is included in the filename for uniqueness and historical tracking as well, just an additional note for clarification.
        csv_filename = None
        if csv_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{csv_prefix}_deals_{timestamp}.csv"
        csv_path = self.to_csv(df, filename=csv_filename)
        results['csv'] = csv_path
        
        # NOTE: Database export
        rows_added = self.to_database(df)
        results['database_rows_added'] = rows_added
        
        # NOTE: Summary statistics as well as showcasing some basic analysis of the data, with min/max/avg prices and counts by category and website.
        # NOTE: handling for nan values in price_numeric column to avoid errors during summary statistics calculation.
        results['summary'] = {
            'total_deals': len(df),
            'deals_with_prices': int(df['price_numeric'].notna().sum()),
            'price_range': {
                'min': float(df['price_numeric'].min()) if not df['price_numeric'].isna().all() else None,
                'max': float(df['price_numeric'].max()) if not df['price_numeric'].isna().all() else None,
                'avg': float(df['price_numeric'].mean()) if not df['price_numeric'].isna().all() else None
            },
            'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            'websites': df['website'].value_counts().to_dict() if 'website' in df.columns else {}
        }
        
        return results

if __name__ == "__main__":
    # NOTE: Production test of the data pipeline has been working for several months now, no issues to report otherwise, occassionally, there is an issue where an item that was webscraped is incorrectly paired, (usually bestbuy) by way of an ad, but otherwise, this is the cleanest solution that I've devised.
    pipeline = DealsDataPipeline()
    print("Data pipeline compiled successfully")