import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "").strip()
TOKEN_URL     = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
OFFERS_URL    = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

HR_STRONG = [
    "drh", "responsable rh", "people ops", "hr manager",
    "chargé rh", "people partner", "engagement", "qvt",
    "bien-être", "expérience collaborateur", "people experience",
    "responsable people", "directeur rh"
]
HR_MEDIUM = [
    "talent", "formation", "learning", "culture", "rh", "people",
    "ressources humaines", "recrutement", "développement rh"
]

VEEP_KEYWORDS = [
    "engagement collaborateur", "engagement des collaborateurs",
    "expérience collaborateur", "culture d'entreprise",
    "bien-être au travail", "qualité de vie au travail", "qvt",
    "satisfaction des employés", "employee engagement",
    "formation", "montée en compétences", "développement des talents",
    "learning", "e-learning", "qhse", "santé au travail",
    "ressources humaines", "people", "drh", "sirh",
    "transformation rh", "digitalisation rh",
]


def _get_token() -> str:
    print(f"CLIENT_ID: {CLIENT_ID[:20]}...")
    print(f"CLIENT_SECRET: {CLIENT_SECRET[:10]}...")
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "scope":         "api_offresdemploiv2 o2dsoffre"
            },
            timeout=10
        )
        print(f"Token status code: {response.status_code}")
        print(f"Token response: {response.text[:300]}")
        return response.json().get("access_token", "")
    except Exception as e:
        print(f"Token exception: {e}")
        return ""
    
def _is_company_match(api_name: str, target: str) -> bool:
    api = api_name.lower()
    tgt = target.lower()
    
    for suffix in ["sas", "sa", "group", "groupe", "holding"]:
        api = api.replace(suffix, "")
        tgt = tgt.replace(suffix, "")

    return tgt.strip() in api.strip()

def _is_company_in_text(offer, company_name):
    text = (
        (offer.get("intitule") or "") +
        (offer.get("description") or "")
    ).lower()

    return company_name.lower() in text

def fetch_job_signals(company_name: str) -> list[dict]:
    """
    Searches France Travail API for HR/People job postings at a given company.
    Falls back to an empty list if API is not configured.
    """
    token = _get_token()
    if not token:
        return _fallback_scrape(company_name)

    try:
        response = requests.get(
            OFFERS_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={
                "motsCles":  " rh ressources humaines recrutement",
                "entreprise": company_name,
                "range":     "0-49",
            },
            timeout=10
        )
        if response.status_code != 200:
            return _fallback_scrape(company_name)

        offers = response.json().get("resultats", [])
        print(f"\n--- DEBUG {company_name} ---")
        print(f"Offers returned: {len(offers)}")

        for o in offers[:5]:
            print(o.get("entreprise", {}).get("nom"))
        jobs   = []

        for offer in offers:
            title   = offer.get("intitule", "")
            url     = offer.get("origineOffre", {}).get("urlOrigine", "")
            company = offer.get("entreprise", {}).get("nom", company_name)
            date    = offer.get("dateCreation", "")

            if company:
                if not _is_company_match(company, company_name):
                    continue
            else:
                if not _is_company_in_text(offer, company_name):
                    continue
            strength = _signal_strength(title)
            if strength > 0:
                jobs.append({
                    "title":           title,
                    "url":             url,
                    "date":            date,
                    "company":         company,
                    "signal_strength": strength
                })
        
        return sorted(jobs, key=lambda x: x["signal_strength"], reverse=True)[:5]

    except Exception as e:
        print(f"[job_scraper] France Travail API error: {e}")
        return _fallback_scrape(company_name)


def _fallback_scrape(company_name: str) -> list[dict]:
    """
    Fallback: scrapes Welcome to the Jungle public search page.
    No API key needed.
    """
    try:
        query = f"{company_name} RH people ressources humaines"
        url      = f"https://www.welcometothejungle.com/fr/jobs?query={query}&page=1"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        soup  = BeautifulSoup(response.text, "html.parser")
        cards = soup.find_all("li", {"data-testid": "search-results-list-item-wrapper"})[:10]
        jobs  = []

        for card in cards:
            title_el = card.find("h3")
            link_el  = card.find("a", href=True)
            if not title_el:
                continue
            title    = title_el.get_text(strip=True)
            href     = link_el["href"] if link_el else ""
            full_url = f"https://www.welcometothejungle.com{href}" if href else ""
            strength = _signal_strength(title)
            if strength > 0:
                jobs.append({
                    "title":           title,
                    "url":             full_url,
                    "date":            "",
                    "company":         company_name,
                    "signal_strength": strength
                })

        return sorted(jobs, key=lambda x: x["signal_strength"], reverse=True)[:5]
    except Exception as e:
        print(f"[job_scraper] Fallback scrape error: {e}")
        return []


def _signal_strength(job_title: str) -> int:
    title_lower = job_title.lower()
    for kw in HR_STRONG:
        if kw in title_lower:
            return 2
    for kw in HR_MEDIUM:
        if kw in title_lower:
            return 1
    return 0


def scrape_website_signals(company_name: str, website_url: str = None) -> dict:
    """
    Scrapes a company website and looks for VEEP-relevant keywords.
    """
    url = website_url or _guess_url(company_name)
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
        )
        if response.status_code != 200:
            return {"url": url, "found": [], "boost": 0, "status": "unreachable"}

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text  = soup.get_text(separator=" ").lower()
        found = [kw for kw in VEEP_KEYWORDS if kw in text]
        boost = min(len(found) * 8, 30)

        return {"url": url, "found": found, "boost": boost, "status": "ok"}

    except Exception as e:
        return {"url": url, "found": [], "boost": 0, "status": f"error: {e}"}


import requests 

def _url_exists(url: str, timeout: int = 3) -> bool:
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except requests.RequestException:
        return False
    
def _guess_url(company_name: str) -> str:
    """
    Guesses the most likely website URL from a company name.
    Simple heuristic — works for most French companies.
    """
    slug = company_name.lower().strip()
    slug = slug.replace(" ", "").replace("-", "").replace("'", "")
    url_fr = f"https://www.{slug}.fr" 
    if _url_exists(url_fr):
        return url_fr

    url_com = f"https://www.{slug}.com"
    if _url_exists(url_com):
        return url_com

    return None

