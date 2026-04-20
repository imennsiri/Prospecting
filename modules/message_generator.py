"""
Message generator
Uses Google Gemini Flash (free tier) to generate personalized
first-touch outreach messages in French for VEEP.

Setup:
1. Go to https://aistudio.google.com/app/apikey
2. Create a free API key
3. Add it to your .env file as: GEMINI_API_KEY=your_key_here
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

VEEP_CONTEXT = """
VEEP est une application mobile native AI dédiée à l'engagement des employés.
Elle contribue à renforcer l'engagement en tenant les équipes informées des 
actualités et des mises à jour de l'entreprise, et en leur donnant le sentiment 
d'être plus connectées à l'organisation. Les employés peuvent accéder à des 
contenus de formation courts, générés par l'IA et adaptés à leur rôle et à 
leurs besoins
VEEP comprend un assistant IA qui peut être connecté à votre documentation 
interne. Les employés peuvent poser des questions et obtenir des réponses 
rapides chaque fois qu'ils ont besoin d'aide. L'application intègre également 
des services RH utiles, tels que l'accès aux fiches de paie, aux informations 
sur les assurances et aux demandes simples comme les congés, le tout depuis
leur téléphone.

Cible idéale : entreprises françaises de 100 à 2000 employés qui investissent
activement dans leur culture RH et l'expérience collaborateur.
"""

def generate_message(
    company_name:  str,
    news_signals:  list[dict],
    score:         int
) -> str:
    """
    Generates a personalized French outreach message for a prospect.
    Falls back to a template if API key is not set.
    """
    if not GEMINI_API_KEY:
        return _fallback_message(company_name, news_signals)

    context = _build_context(company_name, news_signals, score)

    prompt = f"""
Tu es un business developer chez VALUE, une entreprise tunisienne de conseil en transformation digitale
et technologies de l’information.
Tu dois écrire un message de prospection LinkedIn (ou email) court et personnalisé
pour {company_name}, en français.

Contexte sur VEEP:
{VEEP_CONTEXT}

Signaux détectés sur {company_name}:
{context}

Règles:
- Maximum 5 phrases
- Commence par une accroche personnalisée qui montre que tu as fait tes recherches
- Mentionne un signal concret (une actualité ou un poste ouvert)
- Pose une question ouverte à la fin
- Ton professionnel mais humain, pas commercial
- NE PAS mentionner le score ou les données internes
- Signe "Imen, Business Developer chez VALUE"

Écris uniquement le message, sans introduction ni commentaire.
"""

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 300, "temperature": 0.7}
            },
            timeout=15
        )
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        print(f"[message_generator] Gemini error: {e}")
        return _fallback_message(company_name, news_signals)


def _build_context(company_name, news_signals, score) -> str:
    lines = []

    if news_signals:
        lines.append("Actualités récentes:")
        for n in news_signals[:2]:
            lines.append(f"  - {n['title']}")

    return "\n".join(lines) if lines else "Aucun signal spécifique détecté."


def _fallback_message(company_name, news_signals) -> str:
    """
    Simple template used when no API key is configured.
    """
    signal = ""
    if news_signals and news_signals[0]["relevance"] > 0:
        signal = f"J'ai vu que {company_name} était récemment dans l'actualité — {news_signals[0]['title'][:80]}. "

    return (
        f"Bonjour,\n\n"
        f"{signal}"
        f"Je travaille chez VALUE et nous accompagnons des entreprises comme {company_name} "
        f"à mieux mesurer et améliorer l'engagement de leurs collaborateurs grâce à VEEP, "
        f"notre application mobile IA.\n\n"
        f"Est-ce que l'engagement collaborateur est un sujet sur lequel vous travaillez en ce moment ?\n\n"
        f"Imen, Business Developer chez VALUE"
    )
