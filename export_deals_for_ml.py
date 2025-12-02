"""
Export Deals from Database for ML Training
===========================================

This script exports ALL deals from your SQLite database (deals.db) 
into a CSV file ready for Google Colab ML training.

Usage:
    python export_deals_for_ml.py

Output:
    output/ml_training_data file that is used for training ML Models. 


NOTE: Ensure that you have run the scrapers to populate the database before executing this script.
- You could aggregate or add more data later, however, I kept it strictly tied to my webscrapers to demonstrate core functionality of those assets for my projects and the raw power of web scraping in the works.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

def export_deals_for_ml(db_path='output/deals.db', output_dir='output'):
    """
    Export all deals from database into ML training-ready CSV
    """

    print("🔨TECH DEAL FORGE - DATABASE EXPORT FOR ML")
    print("=" * 60)
    
    # Check if database exists
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"\nDatabase not found at {db_path}")
        return None
    
    print(f"\nDatabase: {db_path}")
    print(f"Size: {db_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get total deals count
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM deals")
    total_deals = cursor.fetchone()[0]
    print(f"   Total deals: {total_deals:,}")
    
    if total_deals == 0:
        print(f"\nDatabase is empty!")
        conn.close()
        return None
    
    # Export all deals with required columns
    print(f"\nExporting deals from database...")
    
    query = """
    SELECT 
        title,
        link,
        description,
        price,
        price_numeric,
        price_text,
        original_price,
        discount_percent,
        image_url,
        rating,
        reviews_count,
        availability,
        in_stock,
        website,
        category,
        scraped_at,
        is_active
    FROM deals
    ORDER BY scraped_at DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} deals")
    
    # Data quality report
    print(f"\nData Quality Report:")
    print(f"   - Date range: {df['scraped_at'].min()} → {df['scraped_at'].max()}")
    print(f"   - Websites: {', '.join(df['website'].value_counts().index.tolist())}")
    print(f"   - Categories: {df['category'].nunique()} unique")
    
    print(f"\nPrice Statistics:")
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    valid_prices = df['price_numeric'].dropna()
    if len(valid_prices) > 0:
        print(f"- Min: ${valid_prices.min():.2f}")
        print(f"- Max: ${valid_prices.max():.2f}")
        print(f"- Avg: ${valid_prices.mean():.2f}")
        print(f"- Median: ${valid_prices.median():.2f}")
    
    print(f"\nColumn Completeness:")
    critical_cols = ['price_numeric', 'discount_percent', 'rating', 'reviews_count', 'category']
    for col in critical_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = (missing / len(df)) * 100
            status = "✅" if pct < 20 else "⚠️" if pct < 50 else "❌"
            print(f"   {status} {col}: {pct:.1f}% missing ({missing:,} rows)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'ml_training_data_{timestamp}.csv'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\nExport Complete!")
    print(f"File: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    # Show column list
    print(f"\nExported Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    

    print(f"Select: {output_file.name}")
    print(f"\nML Training in Google Colab can now commence.")
    print()
    
    return output_file

if __name__ == "__main__":
    export_deals_for_ml()
