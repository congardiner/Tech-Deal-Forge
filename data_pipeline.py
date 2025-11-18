import pandas as pd
import sqlite3
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
import math

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class DealsDataPipeline:
    """
    Comprehensive data pipeline for processing scraped deals data.
    Supports CSV, Parquet, SQLite, and MySQL storage with filtering and cleaning.
    """
    
    def __init__(self, output_dir: str = "output", use_mysql: bool = False, use_supabase: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.use_mysql = use_mysql and MYSQL_AVAILABLE
        self.use_supabase = use_supabase and SUPABASE_AVAILABLE
        
        if self.use_mysql:
            # Try to import MySQL configuration
            try:
                from mysql_config import MYSQL_CONFIG
                self.mysql_config = MYSQL_CONFIG.copy()
                self.logger.info("Using MySQL database with configuration from mysql_config.py")
            except ImportError:
                # Fallback to default configuration
                self.mysql_config = {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'dealforge_user',
                    'password': 'dealforge_pass_2024',
                    'database': 'deal_intelligence',
                    'charset': 'utf8mb4'
                }
                self.logger.warning("mysql_config.py not found, using default MySQL configuration")
        elif self.use_supabase:
            # Import Supabase configuration
            try:
                from supabase_config import SUPABASE_CONFIG
                self.supabase_config = SUPABASE_CONFIG.copy()
                self.logger.info("Using Supabase database (cloud PostgreSQL)")
            except ImportError:
                self.logger.error("supabase_config.py not found! Please configure Supabase first.")
                self.use_supabase = False
        else:
            self.db_path = self.output_dir / "deals.db"
            self.logger.info("Using SQLite database")
        
        self._init_database()
    
    def _get_connection(self):
        """Get database connection (SQLite, MySQL, or Supabase)"""
        if self.use_mysql:
            return mysql.connector.connect(**self.mysql_config)
        elif self.use_supabase:
            return psycopg2.connect(**self.supabase_config)
        else:
            return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize database with enhanced schema for rich deal data"""
        if self.use_mysql:
            self._init_mysql_database()
        elif self.use_supabase:
            self._init_supabase_database()
        else:
            self._init_sqlite_database()
    
    def _init_mysql_database(self):
        """Initialize MySQL database and verify connection"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s", 
                      (self.mysql_config['database'],))
        table_count = cursor.fetchone()[0]
        
        if table_count < 5:
            self.logger.warning(f"⚠️ MySQL has only {table_count} tables. Run: python setup_mysql.py")
        else:
            self.logger.info(f"✅ MySQL connected: {table_count} tables | Database: {self.mysql_config['database']}")
        
        conn.close()
    
    def _init_supabase_database(self):
        """Initialize Supabase (PostgreSQL) database and verify connection"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cursor.fetchone()[0]
        
        if table_count < 5:
            self.logger.warning(f"⚠️ Supabase has only {table_count} tables. Run: python migrate_to_supabase.py")
        else:
            self.logger.info(f"✅ Supabase connected: {table_count} tables | Database: {self.supabase_config['database']}")
        
        conn.close()
    
    def _init_sqlite_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    price TEXT,
                    link TEXT UNIQUE,
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
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON deals(price_numeric)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON deals(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_website ON deals(website)")
            
            self.logger.info(f"✅ SQLite database: {self.db_path}")
    
    def clean_data(self, deals: List[Dict]) -> pd.DataFrame:
        """Clean and standardize deals data"""
        df = pd.DataFrame(deals)
        
        if df.empty:
            self.logger.warning("No deals data to clean")
            return df
        
        # Remove duplicates based on link
        original_count = len(df)
        df = df.drop_duplicates(subset=['link'], keep='first')
        removed_dupes = original_count - len(df)
        if removed_dupes > 0:
            self.logger.info(f"Removed {removed_dupes} duplicate deals")
        
        # Clean and standardize titles
        df['title'] = df['title'].str.strip()
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)  # Remove extra whitespace
        
        # Handle price columns - support both old 'price' and new 'price_text'/'price_numeric' format
        if 'price' in df.columns and 'price_text' not in df.columns:
            # Old format (SlickDeals) - extract numeric price from price text
            df['price_numeric'] = df['price'].apply(self._extract_numeric_price)
            df['price_text'] = df['price']  # Keep original price text
        elif 'price_text' in df.columns and 'price_numeric' not in df.columns:
            # New format but missing numeric - extract it
            df['price_numeric'] = df['price_text'].apply(self._extract_numeric_price)
        # If both price_text and price_numeric exist (Newegg format), use as-is
        
        # Clean categories - handle both SlickDeals and Newegg formats
        if 'category' in df.columns:
            # Remove SlickDeals URL prefixes if present
            df['category'] = df['category'].str.replace('https://slickdeals.net/', '', regex=False)
            df['category'] = df['category'].str.replace('/', '', regex=False)
            # Clean up any remaining URL artifacts
            df['category'] = df['category'].str.replace('?', '', regex=False)
            df['category'] = df['category'].str.strip()
        
        # Add scraping timestamp
        df['scraped_at'] = datetime.now().isoformat()
        
        # Filter out deals with very short titles (likely errors)
        df = df[df['title'].str.len() > 10]
        
        self.logger.info(f"Cleaned data: {len(df)} deals ready for processing")
        return df
    
    def _extract_numeric_price(self, price_text: str) -> Optional[float]:
        """Extract numeric price from price text"""
        if not price_text or pd.isna(price_text):
            return None
        
        # Remove currency symbols and extract numbers
        price_clean = re.sub(r'[^\d.,]', '', str(price_text))
        
        if not price_clean:
            return None
        
        # Handle comma-separated thousands
        if ',' in price_clean and '.' in price_clean:
            # Format like 1,299.99
            price_clean = price_clean.replace(',', '')
        elif ',' in price_clean and '.' not in price_clean:
            # Format like 1,299 (no decimal)
            price_clean = price_clean.replace(',', '')
        
        return float(price_clean)
    
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
        
        # Category filter
        if 'categories' in filters and filters['categories']:
            if isinstance(filters['categories'], str):
                filters['categories'] = [filters['categories']]
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Keyword filter in title
        if 'keywords' in filters and filters['keywords']:
            keywords = filters['keywords']
            if isinstance(keywords, str):
                keywords = [keywords]
            
            # Create regex pattern for any keyword
            pattern = '|'.join([re.escape(keyword) for keyword in keywords])
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(pattern, case=False, na=False)
            ]
        
        # Exclude keywords
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
        """Export deals to CSV file"""


        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deals_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Exported {len(df)} deals to CSV: {csv_path}")

        return csv_path
    
    def to_database(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """Save deals to database (SQLite, MySQL, or Supabase)"""
        if self.use_mysql:
            return self._to_mysql_database(df, if_exists)
        elif self.use_supabase:
            return self._to_supabase_database(df, if_exists)
        else:
            return self._to_sqlite_database(df, if_exists)
    
    def _clean_value_for_mysql(self, value):
        """Clean value for MySQL insertion (handle NaN, None, etc.)"""
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, str) and value.lower() in ['nan', 'none', '']:
            return None
        return value
    
    def _to_mysql_database(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """Save deals to MySQL database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            rows_added = 0
            
            for _, row in df.iterrows():
                # Get or create website
                website_id = self._get_or_create_website(cursor, row.get('website', 'slickdeals'), 'mysql')
                
                # Get or create category  
                category_id = self._get_or_create_category(cursor, row.get('category', 'tech'), 'mysql')
                
                # Insert deal
                insert_query = """
                INSERT IGNORE INTO deals (
                    title, link, description, price_text, price_numeric, 
                    original_price, discount_percent, image_url, rating, 
                    reviews_count, availability, in_stock, website_id, 
                    category_id, scraped_at, is_active
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                
                values = (
                    row['title'],
                    row['link'], 
                    self._clean_value_for_mysql(row.get('description')),
                    self._clean_value_for_mysql(row.get('price_text') or row.get('price')),
                    self._clean_value_for_mysql(row.get('price_numeric')),
                    self._clean_value_for_mysql(row.get('original_price')),
                    self._clean_value_for_mysql(row.get('discount_percent')),
                    self._clean_value_for_mysql(row.get('image_url')),
                    self._clean_value_for_mysql(row.get('rating')),
                    self._clean_value_for_mysql(row.get('reviews_count')),
                    self._clean_value_for_mysql(row.get('availability')),
                    row.get('in_stock', True),
                    website_id,
                    category_id,
                    row['scraped_at'],
                    row.get('is_active', True)
                )
                
                cursor.execute(insert_query, values)
                if cursor.rowcount > 0:
                    rows_added += 1
            
            conn.commit()
            self.logger.info(f"Added {rows_added} new deals to MySQL database")
            return rows_added
            
        except Exception as e:
            self.logger.error(f"MySQL insert error: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def _to_supabase_database(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """Save deals to Supabase (PostgreSQL) database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            rows_added = 0
            
            for _, row in df.iterrows():
                # Get or create website
                website_id = self._get_or_create_website(cursor, row.get('website', 'slickdeals'), 'postgres')
                
                # Get or create category  
                category_id = self._get_or_create_category(cursor, row.get('category', 'tech'), 'postgres')
                
                # Insert deal (PostgreSQL syntax - use ON CONFLICT)
                insert_query = """
                INSERT INTO deals (
                    title, link, description, price_text, price_numeric, 
                    original_price, discount_percent, image_url, rating, 
                    reviews_count, availability, in_stock, website_id, 
                    category_id, scraped_at, is_active
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (link) DO NOTHING
                """
                
                values = (
                    row['title'],
                    row['link'], 
                    self._clean_value_for_mysql(row.get('description')),
                    self._clean_value_for_mysql(row.get('price_text') or row.get('price')),
                    self._clean_value_for_mysql(row.get('price_numeric')),
                    self._clean_value_for_mysql(row.get('original_price')),
                    self._clean_value_for_mysql(row.get('discount_percent')),
                    self._clean_value_for_mysql(row.get('image_url')),
                    self._clean_value_for_mysql(row.get('rating')),
                    self._clean_value_for_mysql(row.get('reviews_count')),
                    self._clean_value_for_mysql(row.get('availability')),
                    row.get('in_stock', True),
                    website_id,
                    category_id,
                    row['scraped_at'],
                    row.get('is_active', True)
                )
                
                cursor.execute(insert_query, values)
                if cursor.rowcount > 0:
                    rows_added += 1
            
            conn.commit()
            self.logger.info(f"Added {rows_added} new deals to Supabase database")
            return rows_added
            
        except Exception as e:
            self.logger.error(f"Supabase insert error: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def _get_or_create_website(self, cursor, website_name: str, db_type: str = 'mysql') -> int:
        """Get or create website ID"""
        cursor.execute("SELECT id FROM websites WHERE name = %s", (website_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            # Create new website
            cursor.execute("INSERT INTO websites (name, base_url, is_active) VALUES (%s, %s, %s)", 
                         (website_name, f"https://{website_name}.com", True))
            if db_type == 'postgres':
                # PostgreSQL uses RETURNING clause
                cursor.execute("SELECT lastval()")
                return cursor.fetchone()[0]
            else:
                return cursor.lastrowid
    
    def _get_or_create_category(self, cursor, category_name: str, db_type: str = 'mysql') -> int:
        """Get or create category ID"""
        # Clean category name
        category_clean = category_name.replace('https://slickdeals.net/', '')
        category_clean = category_clean.replace('/', '').replace('?', '').strip()
        
        # Generate slug from category name
        category_slug = category_clean.lower().replace(' ', '-').replace('_', '-')
        # Remove any non-alphanumeric characters except hyphens
        category_slug = re.sub(r'[^a-z0-9\-]', '', category_slug)
        
        cursor.execute("SELECT id FROM categories WHERE name = %s OR slug = %s", (category_clean, category_slug))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            # Create new category with slug
            cursor.execute("INSERT INTO categories (name, slug, description, is_active) VALUES (%s, %s, %s, %s)", 
                         (category_clean, category_slug, f"Auto-created category for {category_clean}", True))
            if db_type == 'postgres':
                # PostgreSQL uses RETURNING clause
                cursor.execute("SELECT lastval()")
                return cursor.fetchone()[0]
            else:
                return cursor.lastrowid
    
    def _to_sqlite_database(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """Save deals to SQLite database - keeps all entries for historical tracking"""
        with sqlite3.connect(self.db_path) as conn:
            # Insert deals, keeping ALL entries for historical perspective
            if if_exists == 'append':
                # Use INSERT to add all deals (no duplicate checking for timeline view)
                rows_added = 0
                for _, row in df.iterrows():
                    cursor = conn.execute("""
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
                        row['category'],
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
                
                self.logger.info(f"Added {rows_added} historical deal entries to SQLite database")
                return rows_added
            else:
                df.to_sql('deals', conn, if_exists=if_exists, index=False)
                self.logger.info(f"Saved {len(df)} deals to SQLite database")
                return len(df)
            



    
    def get_deals_from_db(self, **filters) -> pd.DataFrame:
        """Retrieve deals from database with optional filters"""
        if self.use_mysql:
            return self._get_deals_from_mysql(**filters)
        else:
            return self._get_deals_from_sqlite(**filters)
    
    def _get_deals_from_mysql(self, **filters) -> pd.DataFrame:
        """Retrieve deals from MySQL database with optional filters"""
        conn = self._get_connection()
        
        try:
            query = """
            SELECT d.*, w.name as website_name, c.name as category_name
            FROM deals d
            JOIN websites w ON d.website_id = w.id
            JOIN categories c ON d.category_id = c.id
            WHERE d.is_active = 1
            """
            params = []
            
            if 'min_price' in filters and filters['min_price'] is not None:
                query += " AND (d.price_numeric >= %s OR d.price_numeric IS NULL)"
                params.append(filters['min_price'])
            
            if 'max_price' in filters and filters['max_price'] is not None:
                query += " AND (d.price_numeric <= %s OR d.price_numeric IS NULL)"
                params.append(filters['max_price'])
            
            if 'category' in filters and filters['category']:
                query += " AND c.name = %s"
                params.append(filters['category'])
            
            query += " ORDER BY d.scraped_at DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            self.logger.info(f"Retrieved {len(df)} deals from MySQL database")
            return df
            
        finally:
            conn.close()
    
    def _get_deals_from_sqlite(self, **filters) -> pd.DataFrame:
        """Retrieve deals from SQLite database with optional filters"""
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
        
        query += " ORDER BY scraped_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        self.logger.info(f"Retrieved {len(df)} deals from SQLite database")
        return df
    
    def process_deals(self, deals_data: List[Dict], csv_prefix: Optional[str] = None, **filters) -> Dict[str, Path]:
        """Complete pipeline: clean, filter, and export deals.
        If csv_prefix is provided, CSV will be named like "{prefix}_deals_YYYYMMDD_HHMMSS.csv".
        """
        # Clean the data
        df = self.clean_data(deals_data)
        
        if df.empty:
            self.logger.warning("No deals to process")
            return {}
        
        # Apply filters if provided
        if filters:
            df = self.filter_deals(df, **filters)
        
        # Export to all formats
        results = {}
        
        # CSV export (optionally prefixed to avoid collisions when multiple scrapers run quickly)
        csv_filename = None
        if csv_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{csv_prefix}_deals_{timestamp}.csv"
        csv_path = self.to_csv(df, filename=csv_filename)
        results['csv'] = csv_path
     
        # Database export
        rows_added = self.to_database(df)
        results['database_rows_added'] = rows_added
        
        # Summary statistics
        results['summary'] = {
            'total_deals': len(df),
            'price_range': {
                'min': df['price_numeric'].min() if not df['price_numeric'].isna().all() else None,
                'max': df['price_numeric'].max() if not df['price_numeric'].isna().all() else None,
                'avg': df['price_numeric'].mean() if not df['price_numeric'].isna().all() else None
            },
            'categories': df['category'].value_counts().to_dict(),
            'deals_with_prices': df['price_numeric'].notna().sum()
        }
        
        return results

if __name__ == "__main__":
    # Quick test
    pipeline = DealsDataPipeline(use_mysql=True)
    print("✅ Pipeline initialized successfully")