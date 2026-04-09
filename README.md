# VEEP Prospect Intelligence Tool

Automated research, scoring & outreach generation for French B2B prospects.

## Project structure

```
veep_tool/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── .env.example            # Copy to .env and add your API key
├── data/
│   └── results.csv         # Auto-created when you save prospects
└── modules/
    ├── news_fetcher.py     # Google News RSS scraper
    ├── job_scraper.py      # Indeed FR job postings scraper
    ├── scorer.py           # Fit scoring engine
    ├── message_generator.py # Gemini-powered message generator
    └── tracker.py          # Results CSV tracker
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your Gemini API key (free)
cp .env.example .env
# Edit .env and paste your key from https://aistudio.google.com/app/apikey

# 3. Run the app
streamlit run app.py
```

## Build phases

- [x] Phase 0 — App skeleton (all modules wired, placeholders working)
- [ ] Phase 1 — News fetcher (Google News RSS, keyword scoring)
- [ ] Phase 2 — Job scraper (Indeed FR, HR signal detection)
- [ ] Phase 3 — Fit scoring (tune rules from real outreach data)
- [ ] Phase 4 — Message generator (Gemini Flash prompting)
- [ ] Phase 5 — Feedback loop (results → scoring improvement)
