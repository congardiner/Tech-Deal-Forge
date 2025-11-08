# Tech Deal Forge - Streamlit Dashboard
🔥 **Real-time tech deals aggregator and price tracker**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tech-deal-forge-dashboard.streamlit.app/)
## Features
-  **Price History Tracking** - Track deal prices over time
-  **Price Drop Alerts** - See the biggest discounts
-  **Analytics Dashboard** - Interactive charts and insights
-  **AI Predictions** - ML-powered deal quality scoring
-  **Multi-Source Aggregation** - SlickDeals, Best Buy, and more

## 🚀 Live Demo
Visit the live dashboard: [Tech Deal Forge](https://tech-deal-forge-dashboard.streamlit.app/)

## Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Senior-Project-Tech-Deal-Forge.git
cd Senior-Project-Tech-Deal-Forge

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_dashboard.py
```

## Data Sources
The dashboard displays aggregated data from:
- SlickDeals (Open Global Community Forum)
- Best Buy (Featured Consumer Technology Deals)
- Historical price tracking (SQLite database)

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly
- **Data Processing:** Pandas
- **Database:** SQLite
- **Web Scraping:** Botasaurus + BeautifulSoup
- **ML:** scikit-learn (optional)

## Project Structure
```
Senior-Project-Tech-Deal-Forge/
├── streamlit_dashboard.py      # Main dashboard app
├── slickdeals_webscraper.py    # SlickDeals scraper
├── bestbuy_webscraper.py       # Best Buy scraper
├── data_pipeline.py            # Data processing pipeline
├── output/
│   └── deals.db                # SQLite database
└── requirements.txt            # Python dependencies
```

## 🔄 Updating Data
The dashboard displays cached data that is refreshed every 60 seconds. To manually update:
1. Run scrapers: `python run_all_scrapers.bat`
2. Or use the "🔄 Refresh Data" button in the sidebar

## 📝 License
MIT License - See LICENSE file for details

## 👨‍💻 Author
Created as a Senior Project by Conner Gardiner
