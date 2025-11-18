"""
Simple Streamlit Dashboard for the 'Tech Deal Forge'

My intent with this project has been to streamline the process of keeping things organized, basic, and informative with the data that I've collected.

Hope you enjoy using this Streamlit dashboard and free service for the Tech Deal Forge!

"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import urllib.parse

# NOTE: Issue resolved with path not being resolved. (Local Instance)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "output" / "deals.db"


st.set_page_config(
    page_title="Deal Forge",           
    page_icon="🔨",                     
    layout="wide",                     
    initial_sidebar_state="collapsed"  
)



# -- Category Normalization Section --
# NOTE: This function normalizes noisy/miscategorized category strings into readable labels.
# NOTE: ALL previous issues have been resolved in feeding my categories, as issues kept persisting in scraped input/output for searching and queries. 
# NOTE: RESOLVED.
def normalize_category(raw) -> str:
    """Normalize noisy/miscategorized category strings into readable labels.
    Handles glued prefixes (e.g., 'dealsunlocked'), query tails
    (e.g., '...filters Rating Frontpage'), punctuation, and pluralization.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Other"

    s = str(raw)
    s = urllib.parse.unquote(s)
    # Drop query/fragment of a query from the search state
    s = s.split('?', 1)[0].split('#', 1)[0]
    # Normalize separators
    s = re.sub(r"[\/_>|-]+", " ", s).strip().lower()
    # Remove leading 'deal'/'deals' even if glued or stitched together otherwise.
    s = re.sub(r"^(deals?|deal)(?=[a-z])", "", s)
    # Drop everything after the word 'filters' if present
    s = re.sub(r"\bfilters.*$", "", s)
    # Tokenize and remove stopwords/noise
    tokens = re.findall(r"[a-z]+", s)
    stop = {
        "deal", "deals", "filter", "filters", "rating", "frontpage", "popular",
        "best", "hot", "new", "today", "this", "week", "top", "discount", "sale", "sales",
        "the", "and", "for", "of"
    }
    tokens = [t for t in tokens if t not in stop]
    if not tokens:
        return "Other"
    phrase = " ".join(tokens)

    # Canonical forms for common variations
    canon = {
        # Displays
        "tv": "TVs", "tvs": "TVs", "tv deals": "TVs",
        "video card": "Video Cards", "video cards": "Video Cards", "graphics card": "Graphics Cards",
        # Phones
        "cell phone": "Cell Phones", "cell phones": "Cell Phones",
        "unlocked phone": "Unlocked Phones", "unlocked phones": "Unlocked Phones",
        "smartwatch": "Smartwatches", "smart watch": "Smartwatches", "smart watches": "Smartwatches",
        # Computers
        "laptop": "Laptops", "laptops": "Laptops", "laptop deals": "Laptops",
        "desktop": "Desktops", "desktops": "Desktops", "desktop computers": "Desktops",
        "computer parts": "Computer Parts", "computers parts": "Computer Parts",
        # Components
        "cpu": "CPUs", "cpus": "CPUs", "gpu": "GPUs", "gpus": "GPUs",
        "motherboard": "Motherboards", "motherboards": "Motherboards",
        "memory": "Memory", "ram": "Memory",
        # Audio
        "headphone": "Headphones", "headphones": "Headphones", "wireless headphones": "Wireless Headphones",
        # General
        "tablet": "Tablets", "tablets": "Tablets", "software": "Software",
        "gaming": "Gaming", "electronics": "Electronics", "tech": "Tech", "tech deals": "Tech",
        "drives": "Drives", "education": "Education", "servers": "Servers",
        "video card deals": "Video Cards"
    }
    if phrase in canon:
        return canon[phrase]

    # Title-case fallback, keep specific acronyms uppercased
    title = phrase.title()
    title = re.sub(r"\bCpu(s)?\b", r"CPU\1", title)
    title = re.sub(r"\bGpu(s)?\b", r"GPU\1", title)
    title = re.sub(r"\bTv(s)?\b", r"TV\1", title)
    return title

# SESSION STATE INITIALIZATION
# Initialize all session state variables upfront to prevent re-queries and ghosting
# NOTE: this was a major learning point for me while building this app!


# Data caching in session state (load once per session)
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
    st.session_state.load_timestamp = None

# Filter state (prevents filters from resetting on rerun)
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = []
if 'min_price' not in st.session_state:
    st.session_state.min_price = 0.0
if 'max_price' not in st.session_state:
    st.session_state.max_price = 10000.0
if 'date_range' not in st.session_state:
    st.session_state.date_range = None
if 'row_limit' not in st.session_state:
    st.session_state.row_limit = 10000


# NOTE: Implemented my logo after creating it using Canva
st.image("images/deal-forge-logo/tech_deal_forge_logo.png", width=150)


st.title("The Tech Deal Forge")               
st.subheader("Empowering Data-Driven Decisions for Tech-Related Deals")    


# Header row with theme toggle
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.markdown("**Welcome to Deal Forge!**")  
    st.caption("The data for this website is compiled from various sources with the intent to streamline the process of being informed while purchasing into related consumer product deals. Provision of Deal Insights are provided as a free service, and do not constitute official financial advice.")
    st.caption("Happy Deal Hunting!")
st.markdown("---")


# NOTE: something that I learned, st.cache_data is the way to go for making streamlit super fast to use!
# @st.cache_data decorator prevents reloading data on every interaction

# NOTE: this hasn't been implemented yet ... Still a work in progress, concerns with viability and security.
# Initialize database for cloud deployment (if needed)
if not DB_PATH.exists():
    # Removed try/except: perform a guarded import and fail fast with clear instructions.
    st.error("Database file missing: run your scrapers or initialization script first.")
    st.code(f"Expected path: {DB_PATH}")
    # Optional guarded import (will only run if module present)
    if (BASE_DIR / 'init_cloud_db.py').exists():
        import importlib
        st.info("Attempting one-time initialization via init_cloud_db.init_cloud_database()")
        module = importlib.import_module('init_cloud_db')
        module.init_cloud_database()
        st.success("Database initialized. Please rerun the app.")
    st.stop()

@st.cache_data(ttl=300)
def load_deals_data(row_limit: int = 10000):
    """Load data from SQLite database with caching.
    
    row_limit: maximum rows to return. If <= 0, returns all rows (may be slow).
    """
    conn = sqlite3.connect(str(DB_PATH))
    
    if row_limit and row_limit > 0:
        query = f"SELECT * FROM deals ORDER BY scraped_at DESC LIMIT {int(row_limit)}"
    else:
        query = "SELECT * FROM deals ORDER BY scraped_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
    # Create cleaned category column for consistent filtering/grouping
    if 'category' in df.columns:
        df['category_clean'] = df['category'].apply(normalize_category)
    
    return df

# NOTE: This has ensured that my database has been populated, queries uploaded, and that there are no active issues to report outside of the baseline.
# Load data into session state (only once or when refreshed)
if st.session_state.df_loaded is None:
    with st.spinner("Loading deals data..."):
        st.session_state.df_loaded = load_deals_data(st.session_state.row_limit)
        st.session_state.load_timestamp = datetime.now()

df = st.session_state.df_loaded




if df.empty:
    st.error("No data found! Run your scraper first:")
    st.code("python scraper_with_pipeline.py --format database")
    st.stop()


# NOTE: This is my main layout and the initial gate keeping to my columnns and overall design philosophy.
# MAIN METRICS ROW 
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Deals", len(df))

with col2:
    deals_with_prices = df['price_numeric'].notna().sum()
    st.metric("With Prices", deals_with_prices)

with col3:
    avg_price = df['price_numeric'].mean()
    if pd.notna(avg_price):
        st.metric("Avg Price", f"${avg_price:.2f}")
    else:
        st.metric("Avg Price", "N/A")

with col4:
    if 'scraped_at' in df.columns:
        latest = df['scraped_at'].max()
        st.metric("Last Update", latest.strftime("%m/%d %H:%M"))

st.markdown("---")

# UNIFIED FILTER SECTION
# NOTE: I opted to just have an all-in-one filter section at the top for simplicity
# as otherwise the widgets can overlay one another and make it so that items don't load or adverse ghosting occurs otherwise.

st.subheader("🔍 Search & Filter Deals")
st.caption("Filters apply to all tabs below. Adjust criteria to narrow your search.")

filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

with filter_col1:
    # Keyword search
    search_query = st.text_input(
        "🔎 Keyword Search",
        value=st.session_state.search_query,
        placeholder="e.g., laptop, AMD, monitor...",
        help="Search in deal titles",
        key="main_search_input"
    )
    st.session_state.search_query = search_query

with filter_col2:
    # Category multiselect
    if 'category_clean' in df.columns:
        categories = sorted([c for c in df['category_clean'].dropna().unique().tolist() if c])
    else:
        categories = sorted([c for c in df.get('category', pd.Series([], dtype=str)).dropna().unique().tolist() if c])
    
    selected_categories = st.multiselect(
        "📂 Categories",
        options=categories,
        default=st.session_state.selected_categories,  # empty by default
        help="Select one or more categories",
        key="main_category_select"
    )
    st.session_state.selected_categories = selected_categories

with filter_col3:
    # Data scope selector
    row_limit_option = st.selectbox(
        "� Data Scope",
        options=["10k rows", "20k rows", "All (slow)"],
        index=0,
        help="Limit rows for performance",
        key="main_row_limit_select"
    )
    
    # NOTE: This ensures that the ghosting effect that is common within streamlit isn't continually replicated across sessions, as this has been a redundant issue in scaling out this serviceable site. 
    # EDIT NOTE: no further issues to address in this respect after testing and using caching to improve performance.
    
    if st.button("🔄 Refresh", help="Reload data from database", key="main_refresh_button"):
        # Clear session state data
        st.session_state.df_loaded = None
        st.cache_data.clear()
        # Update row limit if changed
        if row_limit_option.startswith("10k"):
            st.session_state.row_limit = 10000
        elif row_limit_option.startswith("20k"):
            st.session_state.row_limit = 20000
        else:
            st.session_state.row_limit = 0
        st.rerun()

# Second row of filters
filter_col4, filter_col5, filter_col6 = st.columns(3)

with filter_col4:
    # Price range
    if df['price_numeric'].notna().any():
        min_price_data = float(df['price_numeric'].min())
        max_price_data = float(df['price_numeric'].max())
        
        # Handle case where min = max (only one price in dataset)
        if min_price_data == max_price_data:
            st.info(f"💰 Single Price: ${min_price_data:.2f}")
            price_range = (min_price_data, max_price_data)
            st.session_state.min_price = min_price_data
            st.session_state.max_price = max_price_data
        else:
            price_range = st.slider(
                "💰 Price Range ($)",
                min_value=0.0,
                max_value=max_price_data * 1.1,
                value=(st.session_state.min_price, min(st.session_state.max_price, max_price_data)),
                step=10.0,
                key="main_price_slider"
            )
            st.session_state.min_price = price_range[0]
            st.session_state.max_price = price_range[1]
    else:
        st.info("No price data available")
        price_range = (0, 0)

with filter_col5:
    # Date range filter
    if 'scraped_at' in df.columns and df['scraped_at'].notna().any():
        min_dt = pd.to_datetime(df['scraped_at']).min().date()
        max_dt = pd.to_datetime(df['scraped_at']).max().date()
        default_start = max_dt - pd.Timedelta(days=7)
        
        date_range = st.date_input(
            "📅 Date Range",
            value=(default_start.date() if hasattr(default_start, 'date') else default_start, max_dt),
            min_value=min_dt,
            max_value=max_dt,
            help="Filter deals by scrape date",
            key="main_date_range"
        )
        st.session_state.date_range = date_range
    else:
        date_range = None

with filter_col6:
    st.caption(f"**Loaded:** {st.session_state.load_timestamp.strftime('%H:%M:%S') if st.session_state.load_timestamp else 'N/A'}")
    st.caption(f"**Rows:** {len(df):,}")
    if st.session_state.search_query or bool(st.session_state.selected_categories):
        st.caption("🔍 *Filters active*")

st.markdown("---")


# NOTE: APPLIES all filters to create filtered_df (single source of truth) - to ensure that categories are consistent across all tabs
filtered_df = df.copy()

# Apply keyword search
if st.session_state.search_query:
    mask = filtered_df['title'].str.contains(st.session_state.search_query, case=False, na=False)
    filtered_df = filtered_df[mask]

# Apply category filter
if st.session_state.selected_categories:
    col_to_use = 'category_clean' if 'category_clean' in filtered_df.columns else 'category'
    filtered_df = filtered_df[filtered_df[col_to_use].isin(st.session_state.selected_categories)]

# Apply price range
if price_range[0] > 0 or price_range[1] < max_price_data:
    price_mask = (
        ((filtered_df['price_numeric'] >= price_range[0]) & (filtered_df['price_numeric'] <= price_range[1])) |
        filtered_df['price_numeric'].isna()
    )
    filtered_df = filtered_df[price_mask]

# Apply date range
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if 'scraped_at' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['scraped_at'] >= start_ts) & (filtered_df['scraped_at'] <= end_ts)]

# Show filtered count
st.info(f"📊 **{len(filtered_df)} deals** match your filters (from {len(df)} total)")

st.markdown("---")

# ===== TABS SECTION (now using filtered_df as single source) =====
st.header("📊 Analytics & Insights")

# Optional focused deal selection (applies across tabs that support it)
if 'selected_deal_link' not in st.session_state:
    st.session_state.selected_deal_link = None

focus_df = filtered_df[['title','link']].dropna().drop_duplicates()
if not focus_df.empty:
    # Limit options for performance
    max_focus = 500
    focus_df = focus_df.head(max_focus)
    title_to_link = {row.title: row.link for row in focus_df.itertuples()}
    focus_choice = st.selectbox(
        "🎯 Focus on a specific deal (optional)",
        options=["(None)"] + list(title_to_link.keys()),
        index=0,
        help="Select a deal to tailor price history and context."
    )
    if focus_choice != "(None)":
        st.session_state.selected_deal_link = title_to_link[focus_choice]
        st.markdown(f"**Focused Deal:** {focus_choice[:80]}  |  [Open Deal]({st.session_state.selected_deal_link})")
    else:
        st.session_state.selected_deal_link = None
else:
    st.session_state.selected_deal_link = None

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Price History", 
    "📊 Deal vs Average", 
    "🔥 Price Drops",
    "📅 Timeline",
    "🤖 AI Predictions"
])

# Helper function to find deals with price history from a given dataframe
@st.cache_data(ttl=600)
def get_deals_with_history_from_links(links: tuple):
    """Given a tuple of links, return those with multiple price points (history).
    Using tuple for hashability in cache.
    """
    if not links:
        return pd.DataFrame(columns=["link", "title", "entry_count"])
    
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ','.join(['?' for _ in links])
    query = f"""
        SELECT d1.link, d1.title, COUNT(*) as entry_count
        FROM deals d1
        WHERE d1.price_numeric IS NOT NULL 
            AND d1.link IN ({placeholders})
        GROUP BY d1.link
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 100
    """
    results = pd.read_sql(query, conn, params=links)
    conn.close()
    return results

with tab1:
    st.subheader("💰 Price History Tracker")
    st.caption("Shows deals from your current filters that have multiple price snapshots over time")
    
    # Use focused deal if chosen; else discover deals with history
    if st.session_state.selected_deal_link:
        candidate_links = (st.session_state.selected_deal_link,)
    else:
        candidate_links = tuple(filtered_df['link'].dropna().unique()[:500])

    if not candidate_links:
        st.warning("No deals match your current filters. Adjust filters above to see price history.")
    else:
        with st.spinner("Finding deals with price history..."):
            matching_deals = get_deals_with_history_from_links(candidate_links)

        content_container = st.container()
        with content_container:
            if not matching_deals.empty:
                st.success(f"Found {len(matching_deals)} deals with price history in your filtered results")
                # Allow manual selection
                deal_titles_hist = matching_deals['title'].tolist()
                selected_deal_title = st.selectbox(
                    "Select a deal for detailed price history",
                    options=deal_titles_hist,
                    index=0,
                    key="price_history_select"
                )
                st.dataframe(
                    matching_deals[['title','entry_count']]
                        .rename(columns={'title':'Title','entry_count':'Times Tracked'}),
                    hide_index=True,
                )
                deal_link = matching_deals[matching_deals['title'] == selected_deal_title]['link'].iloc[0]
                st.markdown(f"🔗 [Open Deal]({deal_link})")
            else:
                st.info("📊 No deals in your filtered results have multiple price points yet. Run scrapers over multiple days to build history.")
                selected_deal_title = None
                deal_link = None

            if selected_deal_title and deal_link:
                
                conn = sqlite3.connect(str(DB_PATH))
                history_query = """
                    SELECT scraped_at, price_numeric, title, price_text, website
                    FROM deals
                    WHERE link = ?
                    ORDER BY scraped_at ASC
                """
                
                deal_timeline = pd.read_sql(history_query, conn, params=[deal_link])
                deal_timeline['scraped_at'] = pd.to_datetime(deal_timeline['scraped_at'])
                
                conn.close()
                
                # Create line chart showing price over time
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=deal_timeline['scraped_at'],
                    y=deal_timeline['price_numeric'],
                    mode='lines+markers',
                    name='Price',
                    line=dict(color='#FF4B4B', width=3),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                ))

                # Add 7-day moving average (time-based rolling) for smoother context
                try:
                    dt_sorted = deal_timeline.sort_values('scraped_at').copy()
                    dt_sorted = dt_sorted.set_index('scraped_at')
                    dt_sorted['ma7'] = dt_sorted['price_numeric'].rolling('7D').mean()
                    dt_sorted = dt_sorted.reset_index()
                    fig.add_trace(go.Scatter(
                        x=dt_sorted['scraped_at'],
                        y=dt_sorted['ma7'],
                        mode='lines',
                        name='7-day avg',
                        line=dict(color='#2E86AB', width=2, dash='dot'),
                        hovertemplate='<b>Date:</b> %{x}<br><b>7d Avg:</b> $%{y:.2f}<extra></extra>'
                    ))
                except Exception:
                    pass
                
                # Add min/max price annotations
                min_price_idx = deal_timeline['price_numeric'].idxmin()
                max_price_idx = deal_timeline['price_numeric'].idxmax()
                
                fig.add_annotation(
                    x=deal_timeline.loc[min_price_idx, 'scraped_at'],
                    y=deal_timeline.loc[min_price_idx, 'price_numeric'],
                    text=f"Lowest: ${deal_timeline.loc[min_price_idx, 'price_numeric']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='green',
                    bgcolor='green',
                    font=dict(color='white')
                )
                
                fig.add_annotation(
                    x=deal_timeline.loc[max_price_idx, 'scraped_at'],
                    y=deal_timeline.loc[max_price_idx, 'price_numeric'],
                    text=f"Highest: ${deal_timeline.loc[max_price_idx, 'price_numeric']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='red',
                    bgcolor='red',
                    font=dict(color='white')
                )
                
                # Calculate price change
                if len(deal_timeline) > 1:
                    price_change = deal_timeline['price_numeric'].iloc[-1] - deal_timeline['price_numeric'].iloc[0]
                    price_change_pct = (price_change / deal_timeline['price_numeric'].iloc[0]) * 100
                    
                    change_color = "green" if price_change < 0 else "red"
                    change_symbol = "📉" if price_change < 0 else "📈"
                    
                    fig.add_annotation(
                        text=f"{change_symbol} Overall Change: ${price_change:.2f} ({price_change_pct:+.1f}%)",
                        xref="paper", yref="paper",
                        x=0.5, y=1.1,
                        showarrow=False,
                        font=dict(size=16, color=change_color),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor=change_color,
                        borderwidth=2
                    )
                
                fig.update_layout(
                    title=f"Price History: {selected_deal_title[:60]}...",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=450,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=30, label="30d", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]
                        ),
                        rangeslider=dict(visible=True),
                        type='date'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show price statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    current_price = deal_timeline['price_numeric'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    starting_price = deal_timeline['price_numeric'].iloc[0]
                    st.metric("Starting Price", f"${starting_price:.2f}")
                
                with col3:
                    lowest = deal_timeline['price_numeric'].min()
                    st.metric("Lowest Price", f"${lowest:.2f}", 
                             delta=f"{((lowest - current_price) / current_price * 100):.1f}%")
                
                with col4:
                    highest = deal_timeline['price_numeric'].max()
                    st.metric("Highest Price", f"${highest:.2f}",
                             delta=f"{((highest - current_price) / current_price * 100):.1f}%")
                
                with col5:
                    avg_price = deal_timeline['price_numeric'].mean()
                    st.metric("Average Price", f"${avg_price:.2f}")
                
                # Show recent price history table
                st.subheader("📋 Recent Price Changes")
                recent_data = deal_timeline[['scraped_at', 'price_text', 'price_numeric', 'website']].tail(10).sort_values('scraped_at', ascending=False)
                recent_data['scraped_at'] = recent_data['scraped_at'].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(recent_data, hide_index=True)

with tab2:
    st.subheader("📈 Deal vs Category Average")
    st.caption("Compare individual deal prices to their category average (using filtered data)")
    
    # Calculate category averages from filtered_df using cleaned categories
    group_col = 'category_clean' if 'category_clean' in filtered_df.columns else 'category'
    category_avg = (
        filtered_df[filtered_df['price_numeric'].notna()]
        .groupby(group_col)['price_numeric']
        .agg(['mean', 'count'])
        .reset_index()
    )
    category_avg.columns = ['category', 'avg_price', 'deal_count']
    category_avg = category_avg[category_avg['deal_count'] >= 3]  # Only categories with 3+ deals
    
    if not category_avg.empty:
        st.success(f"Found {len(category_avg)} categories with 3+ deals in your filtered results")
        
        # Show as static table
        display_cats = category_avg[['category', 'avg_price', 'deal_count']].copy()
        display_cats['avg_price'] = display_cats['avg_price'].apply(lambda x: f"${x:.2f}")
        display_cats.columns = ['Category', 'Avg Price', 'Deal Count']
        
        st.dataframe(
            display_cats,
            hide_index=True,
        )
        
        # Choose first category automatically
        selected_cat = category_avg.iloc[0]['category']
    else:
        st.info("Not enough data for category comparison in your filtered results. Need at least 3 deals per category.")
        selected_cat = None
    
    if selected_cat:
        # Get deals in this category from filtered_df (using cleaned column when present)
        group_col = 'category_clean' if 'category_clean' in filtered_df.columns else 'category'
        cat_deals = filtered_df[(filtered_df[group_col] == selected_cat) & (filtered_df['price_numeric'].notna())].copy()
        cat_avg_price = category_avg[category_avg['category'] == selected_cat]['avg_price'].iloc[0]
        
        # Sort by date
        cat_deals = cat_deals.sort_values('scraped_at')
        
        # Create comparison chart
        fig = go.Figure()
        
        # Plot individual deals
        fig.add_trace(go.Scatter(
            x=cat_deals['scraped_at'],
            y=cat_deals['price_numeric'],
            mode='markers',
            name='Individual Deals',
            marker=dict(
                size=10, 
                color=cat_deals['price_numeric'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Price ($)"),
                line=dict(width=1, color='white')
            ),
            text=cat_deals['title'].str[:50],
            hovertemplate='<b>%{text}...</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
        ))
        
        # Add average line
        fig.add_hline(
            y=cat_avg_price,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Category Avg: ${cat_avg_price:.2f}",
            annotation_position="right",
            annotation=dict(font=dict(size=14, color="red"))
        )
        
        # Add shaded region for "good deals" (below average)
        fig.add_hrect(
            y0=cat_deals['price_numeric'].min() * 0.95,
            y1=cat_avg_price,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Below Average (Good Deals)",
            annotation_position="left"
        )
        
        fig.update_layout(
            title=f"{selected_cat} - Deals vs Category Average",
            xaxis_title="Date Scraped",
            yaxis_title="Price ($)",
            height=450,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0.05)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        below_avg = cat_deals[cat_deals['price_numeric'] < cat_avg_price]
        above_avg = cat_deals[cat_deals['price_numeric'] >= cat_avg_price]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Deals", len(cat_deals))
        
        with col2:
            st.success(f"🎉 {len(below_avg)} deals below average ({len(below_avg)/len(cat_deals)*100:.1f}%)")
        
        with col3:
            st.warning(f"📊 {len(above_avg)} deals above average ({len(above_avg)/len(cat_deals)*100:.1f}%)")
        
        # Show best deals (below average)
        if not below_avg.empty:
            st.subheader("🏆 Best Deals (Below Average)")
            best_deals = below_avg.nsmallest(5, 'price_numeric')[['title', 'price_numeric', 'website', 'scraped_at']]
            best_deals['scraped_at'] = best_deals['scraped_at'].dt.strftime('%Y-%m-%d')
            best_deals['price_numeric'] = best_deals['price_numeric'].apply(lambda x: f"${x:.2f}")
            st.dataframe(best_deals, hide_index=True)
    else:
        st.info("Not enough data for category comparison. Need at least 3 deals per category.")

# Move function OUTSIDE tab to prevent re-rendering spinner
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_price_drops_from_links(links: tuple):
    """Efficiently compute price drops comparing first-seen vs latest price per link."""
    if not links:
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(DB_PATH))
    # Create placeholders for SQL IN clause
    placeholders = ','.join(['?'] * len(links))
    
    price_drops_query = f"""
        WITH latest AS (
            SELECT d.link, d.title, d.category, d.website, d.price_numeric AS current_price, d.scraped_at AS last_seen
            FROM deals d
            JOIN (
                SELECT link, MAX(scraped_at) AS last_seen
                FROM deals
                WHERE price_numeric IS NOT NULL AND link IN ({placeholders})
                GROUP BY link
            ) m ON d.link = m.link AND d.scraped_at = m.last_seen
            WHERE d.price_numeric IS NOT NULL
        ),
        first AS (
            SELECT d.link, d.price_numeric AS original_price
            FROM deals d
            JOIN (
                SELECT link, MIN(scraped_at) AS first_seen
                FROM deals
                WHERE price_numeric IS NOT NULL AND link IN ({placeholders})
                GROUP BY link
            ) f ON d.link = f.link AND d.scraped_at = f.first_seen
            WHERE d.price_numeric IS NOT NULL
        )
        SELECT 
            l.title,
            l.link,
            l.current_price,
            f.original_price,
            l.category,
            l.website,
            (f.original_price - l.current_price) AS price_drop,
            CASE WHEN f.original_price > 0 THEN ((f.original_price - l.current_price) * 100.0 / f.original_price) ELSE 0 END AS drop_percent,
            l.last_seen
        FROM latest l
        JOIN first f ON l.link = f.link
        WHERE (f.original_price - l.current_price) > 0
        ORDER BY drop_percent DESC
        LIMIT 20
    """
    # Need to pass links twice (for both IN clauses)
    drops = pd.read_sql(price_drops_query, conn, params=links + links)
    conn.close()
    return drops

# ============ ML FEATURE PREPARATION (no try/except; sanitize inputs) ============
def prepare_ml_features(source_df: pd.DataFrame) -> pd.DataFrame:
    df2 = source_df.copy()

    # Ensure datetime
    if 'scraped_at' in df2.columns:
        df2['scraped_at'] = pd.to_datetime(df2['scraped_at'], errors='coerce')
    else:
        df2['scraped_at'] = pd.NaT

    # Base numerics with defaults
    numeric_defaults = {
        'price_numeric': 0.0,
        'discount_percent': 0.0,
        'rating': 3.5,
        'reviews_count': 0,
    }
    for col, default in numeric_defaults.items():
        if col not in df2.columns:
            df2[col] = default
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(default)

    # Website indicators
    website_series = df2.get('website')
    website_l = website_series.astype(str).str.lower() if website_series is not None else pd.Series([''] * len(df2))
    df2['website_bestbuy'] = website_l.str.contains('bestbuy', na=False).astype(int)
    df2['website_slickdeals'] = website_l.str.contains('slickdeals', na=False).astype(int)

    # Category indicators
    category_series = df2.get('category')
    cat_l = category_series.astype(str).str.lower() if category_series is not None else pd.Series([''] * len(df2))
    df2['category_gaming'] = cat_l.str.contains('gam', na=False).astype(int)
    df2['category_laptop'] = cat_l.str.contains('laptop', na=False).astype(int)
    df2['category_monitor'] = cat_l.str.contains('monitor', na=False).astype(int)

    # Temporal features
    dow = df2['scraped_at'].dt.dayofweek.fillna(0).astype(int)
    df2['day_of_week'] = dow
    df2['month'] = df2['scraped_at'].dt.month.fillna(1).astype(int)
    df2['is_weekend'] = dow.isin([5, 6]).astype(int)

    # Relative price features per category
    if 'category' in df2.columns and df2['category'].notna().any():
        cat_group = df2.groupby('category')['price_numeric']
        cat_avg = cat_group.transform('mean').replace(0, pd.NA)
        cat_min = cat_group.transform('min').replace(0, pd.NA)
        df2['price_vs_avg'] = (df2['price_numeric'] / cat_avg).fillna(1.0)
        df2['price_vs_min'] = (df2['price_numeric'] / cat_min).fillna(1.0)
    else:
        df2['price_vs_avg'] = 1.0
        df2['price_vs_min'] = 1.0

    # Times seen (within current set as proxy)
    if 'link' in df2.columns:
        df2['times_seen'] = df2.groupby('link')['link'].transform('count').fillna(1).astype(int)
    else:
        df2['times_seen'] = 1

    # Price std per category (fallback 0) and a neutral recent_trend
    if 'category' in df2.columns:
        df2['price_std'] = df2.groupby('category')['price_numeric'].transform('std').fillna(0.0)
    else:
        df2['price_std'] = 0.0
    df2['recent_trend'] = 0.0

    feature_cols = [
        'price_numeric', 'discount_percent', 'rating', 'reviews_count',
        'website_bestbuy', 'website_slickdeals',
        'category_gaming', 'category_laptop', 'category_monitor',
        'day_of_week', 'month', 'is_weekend',
        'price_vs_avg', 'price_vs_min', 'times_seen', 'price_std', 'recent_trend'
    ]

    # Ensure all features exist
    for c in feature_cols:
        if c not in df2.columns:
            df2[c] = 0

    X = df2[feature_cols].astype(float)
    return X

with tab3:
    st.subheader("🔥 Biggest Price Drops")
    st.caption("Deals that have decreased in price since first scraped (from your filtered results)")
    
    # Get unique links from filtered_df
    candidate_links = tuple(filtered_df['link'].dropna().unique())
    drops = get_price_drops_from_links(candidate_links)
    
    if not drops.empty:
        # Create bar chart of price drops
        fig = go.Figure()
        # Attach customdata with link and full title to support click selection
        fig.add_trace(go.Bar(
            x=drops['drop_percent'],
            y=drops['title'].str[:40] + "...",
            orientation='h',
            marker=dict(
                color=drops['drop_percent'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Discount %")
            ),
            text=drops['drop_percent'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            customdata=drops[['link','title']].values,
            hovertemplate='<b>%{customdata[1]}</b><br>Discount: %{x:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Top 20 Price Drops (Discount %)",
            xaxis_title="Discount Percentage",
            yaxis_title="",
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.05)'
        )
        
        # Show detailed price drop table
        st.subheader("📋 Price Drop Details")
        
        for idx, row in drops.head(10).iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['title'][:60]}...**")
                    try:
                        cat_disp = normalize_category(row.get('category')) if 'category' in row else ''
                    except Exception:
                        cat_disp = row.get('category', '') if isinstance(row, dict) else ''
                    st.caption(f"🏪 {row['website']} | 📂 {cat_disp}")
                
                with col2:
                    st.metric(
                        "Current Price",
                        f"${row['current_price']:.2f}",
                        delta=f"-${row['price_drop']:.2f}",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Discount",
                        f"{row['drop_percent']:.1f}%",
                        delta=f"Was ${row['original_price']:.2f}"
                    )
                
                st.markdown("---")
    else:
        st.info("📊 No price drops detected yet. Run scrapers multiple times to track price changes!")
        st.markdown("""
        **To see price drops:**
        1. Run `run_all_scrapers.bat` daily
        2. Same deals will be tracked over time
        3. Price decreases will appear here!
        """)

with tab4:
    st.subheader("Deals Over Time")
    st.caption("Timeline analysis based on your filtered results")
    
    if 'scraped_at' not in filtered_df.columns or filtered_df['scraped_at'].isna().all():
        st.info("No timestamp data available. Ensure scrapers are storing 'scraped_at'.")
    else:
        # Use filtered_df for timeline
        base_timeline = filtered_df[['scraped_at']].dropna().copy()
        base_timeline['date'] = base_timeline['scraped_at'].dt.date
        daily_counts = base_timeline.groupby('date').size().reset_index(name='deal_count')
        if daily_counts.empty:
            st.info("No date entries to plot.")
        else:
            fig = px.area(daily_counts, x='date', y='deal_count', title='Daily Deal Count Over Time (Filtered)')
            fig.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        # Day-of-week distribution
        base_timeline['day_of_week'] = base_timeline['scraped_at'].dt.day_name()
        dow_counts = base_timeline['day_of_week'].value_counts()
        if not dow_counts.empty:
            st.subheader("📅 Best Days for Deals")
            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dow_counts = dow_counts.reindex([d for d in order if d in dow_counts.index])
            fig = px.bar(x=dow_counts.index, y=dow_counts.values, title='Deals by Day of Week (Filtered)', labels={'x':'Day','y':'Number of Deals'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Hourly pattern (approximate) if hours vary
        if base_timeline['scraped_at'].dt.hour.nunique() > 1:
            st.subheader("🕐 Hourly Deal Pattern")
            hour_counts = base_timeline['scraped_at'].dt.hour.value_counts().sort_index()
            fig = px.line(x=hour_counts.index, y=hour_counts.values, title='Deals by Hour of Day (Filtered)', labels={'x':'Hour (24h)','y':'Number of Deals'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Day vs Hour heatmap for interactive discovery
        try:
            st.subheader("⏱️ Day × Hour Heatmap")
            heat_df = base_timeline.copy()
            heat_df['day'] = heat_df['scraped_at'].dt.day_name()
            heat_df['hour'] = heat_df['scraped_at'].dt.hour
            if not heat_df.empty:
                order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                heat_df['day'] = pd.Categorical(heat_df['day'], categories=order, ordered=True)
                agg = heat_df.groupby(['day','hour']).size().reset_index(name='count')
                fig = px.density_heatmap(
                    agg.sort_values(['day','hour']),
                    x='hour', y='day', z='count',
                    color_continuous_scale='Blues',
                    labels={'hour':'Hour (24h)','day':'Day','count':'Deals'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

with tab5:
    st.title("🤖 AI-Powered Deal Predictions")
    st.caption("Use ML to predict deal quality and find the best deals (from your filtered results)")
    
    # ALWAYS show this message so we know tab5 is rendering
    st.info("🔍 **Debug:** Tab 5 is loading...")
    
    # Model file input
    model_file = st.text_input(
        "📁 Model filename (in project root):",
        value="deal_predictor_20251109_020716.joblib",
        key="ml_model_file"
    )
    
    st.write(f"**Looking for:** `{model_file}`")
    

    # ML Prediction Logic
    # Load model and predict without try/except; inputs are sanitized beforehand
    import joblib
    import os

    if not os.path.exists(model_file):
        st.error(f"❌ Model file not found: `{model_file}`")
        st.info("📋 Make sure the .joblib file is in your project root folder")
        st.code(f"Expected path: {os.path.abspath(model_file)}")
        st.stop()

    st.success("✅ File exists!")
    with st.spinner("Loading model..."):
        model = joblib.load(model_file)
    st.success(f"✅ Model loaded: {type(model).__name__}")

    # Predict for rows with valid prices from filtered_df
    valid_df = filtered_df[(filtered_df['price_numeric'].notna()) & (filtered_df['price_numeric'] > 0)].copy()
    st.info(f"Found {len(valid_df)} deals with prices in your filtered results")

    if len(valid_df) > 0:
        # Row selection / performance cap
        total_rows = int(len(valid_df))
        if total_rows <= 500:
            # Avoid Streamlit slider error when min == max; just score all rows
            st.info(f"Scoring all {total_rows} deals (slider disabled: not enough rows for range)")
            max_rows = total_rows
        else:
            # Dynamic slider: allow selecting between 500 and total_rows
            default_value = min(5000, total_rows)
            step_size = 500 if total_rows > 2000 else 100
            max_rows = st.slider(
                "Max rows to score",
                min_value=500,
                max_value=total_rows,
                value=default_value,
                step=step_size,
                key="ml_max_rows"
            )
        valid_df = valid_df.head(max_rows)

        with st.spinner(f"Predicting {len(valid_df)} deals..."):
            X = prepare_ml_features(valid_df)

            # Validate feature shape against model
            n_in = getattr(model, 'n_features_in_', None)
            if n_in is not None and X.shape[1] != int(n_in):
                st.error(f"Model expects {int(n_in)} features but received {X.shape[1]}.")
                st.code("Provided features:\n" + "\n".join(list(X.columns)))
                st.stop()

            scores = model.predict(X)
            valid_df['ml_score'] = scores

        st.success(f"✅ Predicted {len(valid_df)} deals!")

        # Show stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            excellent = (valid_df['ml_score'] >= 75).sum()
            st.metric("🔥 Excellent", excellent, f"{excellent/len(valid_df)*100:.0f}%")
        with col2:
            good = ((valid_df['ml_score'] >= 60) & (valid_df['ml_score'] < 75)).sum()
            st.metric("👍 Good", good, f"{good/len(valid_df)*100:.0f}%")
        with col3:
            fair = ((valid_df['ml_score'] >= 40) & (valid_df['ml_score'] < 60)).sum()
            st.metric("👌 Fair", fair, f"{fair/len(valid_df)*100:.0f}%")
        with col4:
            poor = (valid_df['ml_score'] < 40).sum()
            st.metric("❌ Poor", poor, f"{poor/len(valid_df)*100:.0f}%")

        # Distribution chart
        st.subheader("📊 Score Distribution")
        fig = px.histogram(valid_df, x='ml_score', nbins=20, title="ML Quality Scores")
        fig.add_vline(x=75, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_vline(x=60, line_dash="dash", line_color="blue", annotation_text="Good")
        fig.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Fair")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Top 10
        st.subheader("🏆 Top 10 Recommended Deals")
        top10 = valid_df.nlargest(10, 'ml_score').copy()

        for _, deal in top10.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{deal.get('title', 'N/A')[:70]}...**")
                st.caption(f"🏪 {deal.get('website', 'N/A')} | 📂 {deal.get('category', 'N/A')}")
            with col2:
                st.metric("Price", f"${deal.get('price_numeric', 0):.2f}")
            with col3:
                score = deal['ml_score']
                if score >= 75:
                    st.success(f"**{score:.1f}**")
                elif score >= 60:
                    st.info(f"**{score:.1f}**")
                else:
                    st.warning(f"**{score:.1f}**")
            if pd.notna(deal.get('link')):
                st.markdown(f"🔗 [View Deal]({deal['link']})")
            st.markdown("---")


st.header("🔍 Deal Search Results")

# Show how many results
st.write(f"**Found {len(filtered_df)} deals** (filtered from {len(df)} total)")

# Select which columns to show
# Prefer cleaned category if present
display_columns = ['title', 'price', 'category_clean', 'scraped_at'] if 'category_clean' in filtered_df.columns else ['title', 'price', 'category', 'scraped_at']
available_columns = [col for col in display_columns if col in filtered_df.columns]

# Show data table
if not filtered_df.empty and available_columns:
    
    # Sort options
    sort_column = st.selectbox(
        "Sort by:",
        options=available_columns,
        index=len(available_columns)-1,  # Default to last column (usually timestamp)
        key="main_sort_column_selectbox"
    )
    
    # Sort the data
    if sort_column in filtered_df.columns:
        display_df = filtered_df[available_columns].sort_values(sort_column, ascending=False)
    else:
        display_df = filtered_df[available_columns]
    
    # Pagination - show 20 deals per page
    page_size = 20
    total_pages = len(display_df) // page_size + 1
    
    # Page selector
    page_number = st.number_input(
        f"Page (1-{total_pages}):",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key="main_page_number"
    )
    
    # Calculate start and end indices
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    
    # NOTE: Displays the data slice as intended, no issues to report after edge cases were handled.
    show_cards = st.checkbox("Show deals as cards", value=False, help="Toggle between table view and card view", key="show_cards_toggle")
    page_slice = display_df.iloc[start_idx:end_idx]
    if show_cards:
        for _, row in page_slice.iterrows():
            title = str(row.get('title',''))
            price = row.get('price') or row.get('price_numeric')
            cat_label = row.get('category_clean') or row.get('category')
            ts = row.get('scraped_at')
            link = row.get('link')
            price_disp = f"${price:.2f}" if isinstance(price,(int,float)) else str(price)
            meta = f"{price_disp} | {cat_label} | {ts}" if ts else f"{price_disp} | {cat_label}"
            st.markdown(f"**{title[:90]}**\n{meta}\n{'🔗 [Open Deal](' + link + ')' if isinstance(link,str) else ''}")
            st.markdown("---")
    else:
        st.dataframe(
            page_slice,
            hide_index=True
        )
    

    # FILE DOWNLOADS for my csv and json outputs. 
    # NOTE: all issues with missing values have been properly handled, no further items to make note of. 
    
    st.subheader("📁 Directly Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        # Automatic CSV conversion
        csv_data = filtered_df.to_csv(index=False)
        

        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"deals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the filtered results as an Excel-Friendly CSV file"
        )
    
    with col2:
        # Convert to JSON
        json_data = filtered_df.to_json(orient='records', indent=2)
        
        st.download_button(
            label="📋 Download as JSON", 
            data=json_data,
            file_name=f"deals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download the filtered results as a JSON file"
        )

else:
    st.info("No deals found matching your current search criteria")


# ===== RETAILER LEADERBOARD =====
st.markdown("---")
st.header("🏪 Retailer Leaderboard")
st.caption("Volume, median price, and average discount (from tracked drops)")

if 'website' in filtered_df.columns:
    # Volume and median price from filtered_df
    vol = (
        filtered_df.groupby('website')
        .agg(volume=('link', 'nunique'), median_price=('price_numeric', 'median'))
        .reset_index()
    )
    # Compute average drop % by website using price drop helper on current filtered links
    links_tuple = tuple(filtered_df['link'].dropna().unique())
    drops_web = get_price_drops_from_links(links_tuple)
    if not drops_web.empty:
        drops_agg = drops_web.groupby('website')['drop_percent'].mean().reset_index().rename(columns={'drop_percent':'avg_drop_percent'})
        leaderboard = vol.merge(drops_agg, on='website', how='left')
    else:
        leaderboard = vol.copy()
        leaderboard['avg_drop_percent'] = float('nan')
    # Format
    leaderboard = leaderboard.sort_values(['volume','median_price'], ascending=[False, True])
    leaderboard['median_price'] = leaderboard['median_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    leaderboard['avg_drop_percent'] = leaderboard['avg_drop_percent'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    st.dataframe(leaderboard.rename(columns={
        'website': 'Website', 'volume':'Unique Deals', 'median_price':'Median Price', 'avg_drop_percent':'Avg Drop %'
    }), hide_index=True, use_container_width=True)
else:
    st.info("Website column not found in data.")

# Footer with status
st.markdown("---")  # Horizontal line

# Status information
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**Database:** {DB_PATH}")

with col2:
    last_loaded = st.session_state.get('load_timestamp', 'Never')
    if last_loaded != 'Never':
        st.write(f"**Data Loaded:** {last_loaded.strftime('%H:%M:%S')}")
    else:
        st.write(f"**Data Loaded:** Never")

with col3:
    st.write(f"**Total Deals:** {len(df)} | **Filtered:** {len(filtered_df)}")

# NOTE: Footer Section
# Utilized a few assets from github including the stock image as seen within my streamlit site, all rights are reserved for Github in the usage of this image, with copyright belonging to them.
# All other comments and included textual context is my own accord and making, just leaving a note of that here though for clarity.

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 26px; background-color: rgba(0,0,0,0.05); border-radius: 10px; margin-top: 20px;">
    <p style="font-size: 18px; margin: 16px 0;">
        <strong>Tech Deal Forge</strong> - Empowering Data-Driven Decisions
    </p>
    <p style="font-size: 16px; margin: 16px 0;">
        <a href="https://github.com/congardiner/Senior-Project---Tech-Deal-Forge" target="_blank" style="text-decoration: none; color: #0366d6;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="26" style="vertical-align: middle; margin-right: 8px;">
            <span style="text-decoration: underline;">View on GitHub</span>
        </a>
    </p>
    <p style="font-size: 16px; color: #666; margin: 12px 0;">
        © 2025 Tech Deal Forge | Licensed under the MIT License
    </p>
    <p style="font-size: 16px; color: #888; margin: 5px 0;">
        This tool is provided for informational purposes only on tech-related product deals and does not constitute financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
