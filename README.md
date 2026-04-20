# VEEP Prospect Intelligence Tool

Automated research, scoring & outreach generation for French B2B prospects.

## Project structure

```
veep_tool/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── data/
│   └── results.csv         # Auto-created when you save prospects
└── modules/
    ├── news_fetcher.py     # Google News RSS scraper
    ├── scorer.py           # Fit scoring engine
    ├── message_generator.py # Gemini-powered message generator


