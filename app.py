import asyncio
import os
import re
from urllib.parse import urlparse

import httpx
import anthropic
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
GMAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


def get_ai_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY mangler — tjek Railway Variables")
    return anthropic.Anthropic(api_key=key)

PLACE_FIELDS = (
    "name,rating,user_ratings_total,opening_hours,"
    "formatted_phone_number,website,formatted_address,"
    "photos,business_status"
)


class AnalyzeRequest(BaseModel):
    url: str


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


async def fetch_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0)"}
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.text


async def get_pagespeed(url: str) -> dict:
    api = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&strategy=mobile"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(api)
            return r.json()
    except Exception:
        return {}


async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    r = await client.get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        params={"place_id": place_id, "fields": PLACE_FIELDS, "key": GMAPS_KEY},
    )
    return r.json().get("result", {})


async def get_google_business(domain: str, page_title: str) -> dict:
    if not GMAPS_KEY:
        return {"found": False}

    clean_domain = domain.replace("www.", "")
    queries = [page_title, clean_domain] if page_title else [clean_domain]

    async with httpx.AsyncClient(timeout=20) as client:
        for query in queries:
            # Try findplacefromtext first (fast, single result)
            r = await client.get(
                "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                params={
                    "input": query,
                    "inputtype": "textquery",
                    "fields": "place_id,website",
                    "key": GMAPS_KEY,
                },
            )
            candidates = r.json().get("candidates", [])
            for c in candidates:
                if clean_domain in c.get("website", ""):
                    details = await get_place_details(client, c["place_id"])
                    return {"found": True, **details}

            # Fallback: textsearch — check top 5 results for website match
            r2 = await client.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={"query": query, "key": GMAPS_KEY},
            )
            results = r2.json().get("results", [])
            for result in results[:5]:
                pid = result["place_id"]
                # Cheap lookup: only fetch website field first
                r3 = await client.get(
                    "https://maps.googleapis.com/maps/api/place/details/json",
                    params={"place_id": pid, "fields": "website", "key": GMAPS_KEY},
                )
                website = r3.json().get("result", {}).get("website", "")
                if clean_domain in website:
                    details = await get_place_details(client, pid)
                    return {"found": True, **details}

    return {"found": False}


def analyze_website(html: str, url: str, pagespeed: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    issues, positives = [], []

    # HTTPS
    if url.startswith("https://"):
        positives.append("Siden bruger HTTPS")
    else:
        issues.append("Siden bruger ikke HTTPS — Google straffer usikre sider")

    # Title
    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    if not title_text:
        issues.append("Mangler sidetitel — kritisk for Google-rangering")
    elif len(title_text) < 20:
        issues.append(f"Sidetitlen er meget kort ({len(title_text)} tegn) — bør være 50–60 tegn")
    elif len(title_text) > 65:
        issues.append(f"Sidetitlen er for lang ({len(title_text)} tegn) — Google kapper den af")
    else:
        positives.append(f"God sidetitel ({len(title_text)} tegn)")

    # Meta description
    meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    desc = meta_desc.get("content", "").strip() if meta_desc else ""
    if not desc:
        issues.append("Mangler meta-beskrivelse — den tekst Google viser i søgeresultater")
    elif len(desc) < 100:
        issues.append(f"Meta-beskrivelsen er for kort ({len(desc)} tegn) — bør være 150–160 tegn")
    elif len(desc) > 165:
        issues.append(f"Meta-beskrivelsen er for lang ({len(desc)} tegn)")
    else:
        positives.append("Har god meta-beskrivelse")

    # H1
    h1s = soup.find_all("h1")
    if not h1s:
        issues.append("Mangler H1-overskrift — vigtigt signal til Google")
    elif len(h1s) > 1:
        issues.append(f"Har {len(h1s)} H1-overskrifter — bør kun have én")
    else:
        positives.append("Har én H1-overskrift")

    # LocalBusiness schema
    schemas = soup.find_all("script", type="application/ld+json")
    has_local = any(
        s.string and any(t in s.string for t in ["LocalBusiness", "Organization", "Contractor"])
        for s in schemas
    )
    if has_local:
        positives.append("Har LocalBusiness schema markup")
    else:
        issues.append("Mangler LocalBusiness schema markup — afgørende for lokal SEO")

    # Telefonnummer
    if re.search(r"(\+45[\s\-]?)?[2-9]\d[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}", text):
        positives.append("Telefonnummer synligt på siden")
    else:
        issues.append("Intet telefonnummer fundet — Google bruger NAP til lokal rangering")

    # Adresse
    if any(kw in text.lower() for kw in ["vej", "gade", "allé", "plads", "stræde", "boulevard", "torv"]):
        positives.append("Adresse fundet på siden")
    else:
        issues.append("Ingen fysisk adresse — svækker lokal SEO markant")

    # Google Maps embed
    if "google.com/maps" in html or "maps.googleapis.com" in html:
        positives.append("Google Maps integreret på siden")
    else:
        issues.append("Ingen Google Maps embed — forbedrer lokal troværdighed")

    # Mobilvenlig
    if soup.find("meta", attrs={"name": re.compile("viewport", re.I)}):
        positives.append("Mobilvenlig (viewport-tag)")
    else:
        issues.append("Sandsynligvis ikke mobilvenlig — 60%+ af lokale søgninger er på mobil")

    # Alt-tekst på billeder
    imgs = soup.find_all("img")
    no_alt = [i for i in imgs if not i.get("alt", "").strip()]
    if imgs and len(no_alt) > len(imgs) * 0.4:
        issues.append(f"{len(no_alt)} af {len(imgs)} billeder mangler alt-tekst")

    # Open Graph
    if soup.find("meta", property="og:title"):
        positives.append("Har Open Graph tags")
    else:
        issues.append("Mangler Open Graph tags — siden ser dårlig ud delt på sociale medier")

    # PageSpeed
    try:
        score = int(pagespeed["lighthouseResult"]["categories"]["performance"]["score"] * 100)
        if score < 50:
            issues.append(f"Meget langsom på mobil (PageSpeed: {score}/100)")
        elif score < 75:
            issues.append(f"Langsom på mobil (PageSpeed: {score}/100)")
        else:
            positives.append(f"God mobilhastighed (PageSpeed: {score}/100)")
    except (KeyError, TypeError):
        pass

    return {"issues": issues, "positives": positives, "title": title_text}


def analyze_gbp(gbp: dict, domain: str) -> tuple[list, list, dict]:
    issues, positives = [], []
    summary = {}

    if not gbp.get("found"):
        issues.append("INGEN Google Business Profil fundet — I er usynlige på Google Maps og i lokale søgninger")
        summary["status"] = "not_found"
        return issues, positives, summary

    summary["status"] = "found"
    summary["name"] = gbp.get("name", "")
    summary["address"] = gbp.get("formatted_address", "")
    summary["phone"] = gbp.get("formatted_phone_number", "")

    business_status = gbp.get("business_status", "")
    if business_status == "CLOSED_TEMPORARILY":
        issues.append("Google Business viser virksomheden som midlertidigt lukket")
    elif business_status == "CLOSED_PERMANENTLY":
        issues.append("Google Business viser virksomheden som PERMANENT LUKKET")

    # Anmeldelser
    rating = gbp.get("rating")
    reviews = gbp.get("user_ratings_total", 0)
    summary["rating"] = rating
    summary["reviews"] = reviews

    if reviews == 0:
        issues.append("0 Google-anmeldelser — anmeldelser er afgørende for lokal rangering og kundernes tillid")
    elif reviews < 5:
        issues.append(f"Kun {reviews} Google-anmeldelse(r) — for få til at påvirke rangeringen")
    elif reviews < 20:
        issues.append(f"Kun {reviews} Google-anmeldelser — konkurrenterne har sandsynligvis flere")
    else:
        positives.append(f"{reviews} Google-anmeldelser")

    if rating is not None:
        if rating < 3.5:
            issues.append(f"Lav Google-rating: {rating}/5 — skræmmer kunder væk")
        elif rating < 4.2:
            issues.append(f"Middel Google-rating: {rating}/5 — kan forbedres markant")
        else:
            positives.append(f"God Google-rating: {rating}/5")

    # Åbningstider
    if gbp.get("opening_hours"):
        positives.append("Åbningstider udfyldt på Google Business")
        summary["has_hours"] = True
    else:
        issues.append("Åbningstider mangler på Google Business — kunder ved ikke hvornår I er åbne")
        summary["has_hours"] = False

    # Telefonnummer
    if gbp.get("formatted_phone_number"):
        positives.append("Telefonnummer på Google Business")
    else:
        issues.append("Intet telefonnummer på Google Business Profil")

    # Billeder
    photos = gbp.get("photos", [])
    photo_count = len(photos)
    summary["photos"] = photo_count
    if photo_count == 0:
        issues.append("Ingen billeder på Google Business — profiler med billeder får 42% flere forespørgsler")
    elif photo_count < 5:
        issues.append(f"Kun {photo_count} billede(r) på Google Business — tilføj flere for bedre synlighed")
    else:
        positives.append(f"{photo_count}+ billeder på Google Business")

    # Website-match
    clean_domain = domain.replace("www.", "")
    gbp_website = gbp.get("website", "")
    if gbp_website and clean_domain not in gbp_website:
        issues.append("Hjemmesiden på Google Business matcher ikke den analyserede URL")

    return issues, positives, summary


def build_pitches(domain: str, site_issues: list, gbp_issues: list, gbp_summary: dict) -> tuple[str, str]:
    all_issues = gbp_issues[:3] + site_issues[:3]
    top_issues = all_issues[:5]
    issues_text = "\n".join(f"- {i}" for i in top_issues) if top_issues else "Ingen store problemer"

    gbp_context = ""
    if gbp_summary.get("status") == "not_found":
        gbp_context = "De har INGEN Google Business Profil."
    elif gbp_summary.get("status") == "found":
        r = gbp_summary.get("rating", "?")
        rv = gbp_summary.get("reviews", 0)
        gbp_context = f"Google Business fundet: {rv} anmeldelser, rating {r}/5."

    prompt = f"""Du er sælger for Håndværkerregistret, som sælger lokal SEO og Google Business-optimering til håndværkere og lokale virksomheder i Danmark.

Du har analyseret: {domain}
{gbp_context}

TOP PROBLEMER FUNDET:
{issues_text}

Skriv to ting:

1. TELEFON-PITCH: En naturlig tekst sælgeren kan læse op (ca. 30-40 sekunder). Den skal:
   - Starte med: "Hej, jeg ringer fra Håndværkerregistret"
   - Nævne 2-3 KONKRETE problemer vi fandt på DERES side/Google profil
   - Forklare hvad det koster dem i mistede kunder (gerne med tal/procenter)
   - Slutte med ét åbent spørgsmål der inviterer til dialog
   - Lyde naturligt og venligt — ikke som en robot eller salgsscript

2. SMS: Max 155 tegn. Kort og direkte. Nævn at vi har kigget på deres side. Ikke spam-agtigt.

Format — brug præcis disse overskrifter:
TELEFON-PITCH:
[tekst]

SMS:
[tekst]"""

    msg = get_ai_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text
    phone, sms = "", ""

    if "TELEFON-PITCH:" in raw and "SMS:" in raw:
        parts = raw.split("SMS:", 1)
        phone = parts[0].replace("TELEFON-PITCH:", "").strip()
        sms = parts[1].strip()
    else:
        phone = raw.strip()
        sms = f"Hej! Vi har analyseret {domain} og fundet vigtige SEO-problemer. Ring til Håndværkerregistret – vi kan hjælpe."

    return phone, sms


@app.post("/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    url = normalize_url(req.url)
    domain = urlparse(url).netloc

    try:
        html = await fetch_page(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kunne ikke hente siden: {e}")

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    pagespeed, gbp = await asyncio.gather(
        get_pagespeed(url),
        get_google_business(domain, page_title),
        return_exceptions=True,
    )
    if isinstance(pagespeed, Exception):
        pagespeed = {}
    if isinstance(gbp, Exception):
        gbp = {"found": False}

    site_result = analyze_website(html, url, pagespeed)
    gbp_issues, gbp_positives, gbp_summary = analyze_gbp(gbp, domain)

    phone, sms = build_pitches(
        domain,
        site_result["issues"],
        gbp_issues,
        gbp_summary,
    )

    return {
        "url": url,
        "domain": domain,
        "site_issues": site_result["issues"],
        "site_positives": site_result["positives"],
        "gbp_issues": gbp_issues,
        "gbp_positives": gbp_positives,
        "gbp_summary": gbp_summary,
        "phone_pitch": phone,
        "sms_pitch": sms,
    }


@app.get("/health")
async def health():
    return {
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gmaps_key_set": bool(os.environ.get("GOOGLE_MAPS_API_KEY")),
        "env_var_names": [k for k in os.environ.keys() if "KEY" in k or "API" in k or "ANTHROPIC" in k],
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
