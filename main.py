from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, time
import requests, requests_cache
from requests.exceptions import HTTPError, RequestException
import yfinance as yf

# -------------------------------------------------------------------
# Cache outbound HTTP for 1 hour (reduces rate-limits)
# -------------------------------------------------------------------
requests_cache.install_cache("http_cache", expire_after=3600)

app = FastAPI(title="SMID Snapshot API (Yahoo→FMP fallback)")

SEC_HEADERS = {"User-Agent": "Craig-FinSnapshot/1.0 (contact: you@example.com)"}
FMP_KEY = os.environ.get("FMP_API_KEY")
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# ---------------------------- Models -------------------------------

class Snapshot(BaseModel):
    ticker: str
    name: str | None = None
    price: float | None = None
    market_cap: float | None = None
    metrics: dict
    composite_score: int
    stars: int
    stars_text: str
    filings: dict
    notes: str | None = None

# ----------------------- Utility functions -------------------------

def stars_from_score(score: int) -> tuple[int, str]:
    for th, s in [(80,5),(60,4),(40,3),(20,2),(0,1)]:
        if score >= th:
            return s, "★"*s + "☆"*(5-s)

def latest_filing(cik: str | int | None) -> str | None:
    if not cik:
        return None
    try:
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        r = requests.get(url, headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json().get("filings", {}).get("recent", {})
        forms = data.get("form", [])
        accs  = data.get("accessionNumber", [])
        for i, form in enumerate(forms):
            if form in ("10-Q", "10-K"):
                acc = accs[i].replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}-index.html"
    except Exception:
        return None
    return None

def score_company(m: dict) -> int:
    score = 0
    pe, peg = m.get("pe_fwd"), m.get("peg_fwd")
    score += 20 if (pe is not None and pe < 18) else 10 if (pe is not None and pe < 25) else 0
    score += 20 if (peg is not None and peg <= 1.0) else 10 if (peg is not None and peg <= 1.5) else 0

    roic = (m.get("roic") or 0) or 0
    opm  = (m.get("op_margin") or 0) or 0
    score += 20 if roic >= 0.12 else 10 if roic >= 0.08 else 0
    score += 10 if opm  >= 0.15 else 5  if opm  >= 0.08 else 0

    rev = (m.get("rev_yoy") or 0) or 0
    eps = (m.get("eps_yoy") or 0) or 0
    score += 15 if rev >= 0.20 else 8 if rev >= 0.12 else 0
    score += 15 if eps >= 0.20 else 8 if eps >= 0.12 else 0

    de = m.get("de_ratio")
    score += 10 if (de is not None and de <= 0.5) else 5 if (de is not None and de <= 1.0) else 0

    return max(0, min(100, score))

# ------------------- Yahoo (primary) with retry --------------------

def fetch_yahoo_info_with_retry(ticker: str, max_retries: int = 4, base_sleep: float = 1.5):
    t = yf.Ticker(ticker)
    last_err = None
    for attempt in range(max_retries):
        try:
            info = t.get_info()  # rich fields; may 429/crumb
            if info:
                return info, t
            last_err = ValueError("Empty info payload")
            raise last_err
        except Exception as e:
            msg = str(e)
            last_err = e
            transient = any(s in msg for s in ("429", "Too Many Requests", "Invalid Crumb", "timed out", "Max retries", "EOF"))
            if attempt < max_retries - 1 and transient:
                time.sleep(base_sleep * (2 ** attempt))  # 1.5, 3, 6, 12...
                continue
            break

    # If still failing, raise specific HTTP for caller to decide
    if "429" in str(last_err) or "Too Many Requests" in str(last_err) or "Invalid Crumb" in str(last_err):
        raise HTTPException(status_code=429, detail="Yahoo blocked the request (rate/crumb).")
    raise HTTPException(status_code=502, detail=f"Yahoo request failed: {last_err}")

# ---------------------- FMP (fallback provider) --------------------

def fmp_get(path: str, params: dict | None = None):
    if not FMP_KEY:
        raise HTTPException(status_code=503, detail="FMP API key not set. Set FMP_API_KEY env var.")
    params = params.copy() if params else {}
    params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}{path}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data

def fetch_fmp_info(ticker: str) -> dict:
    """
    Pulls profile (name, price, mcap, cik), ratios-ttm (margins, PE, D/E),
    and financial-growth (revenue & EPS growth). Returns a dict shaped like Yahoo's 'info'.
    """
    try:
        profile = fmp_get(f"/profile/{ticker}")
        profile = profile[0] if isinstance(profile, list) and profile else {}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FMP profile error: {e}")

    # Ratios TTM for margins/PE/DE
    try:
        ratios = fmp_get(f"/ratios-ttm/{ticker}")
        ratios = ratios[0] if isinstance(ratios, list) and ratios else {}
    except Exception:
        ratios = {}

    # Financial growth (revenueGrowthTTM, epsgrowthTTM)
    try:
        growth = fmp_get(f"/financial-growth/{ticker}")
        growth = growth[0] if isinstance(growth, list) and growth else {}
    except Exception:
        growth = {}

    # Map to a Yahoo-like dict
    info = {
        "longName": profile.get("companyName") or profile.get("symbol"),
        "currentPrice": profile.get("price"),
        "marketCap": profile.get("mktCap"),
        "forwardPE": ratios.get("priceEarningsRatioTTM"),
        "pegRatio": None,  # not directly provided; could compute if you have forward growth
        "grossMargins": ratios.get("grossProfitMarginTTM"),
        "operatingMargins": ratios.get("operatingProfitMarginTTM"),
        "revenueGrowth": growth.get("revenueGrowthTTM"),
        "earningsGrowth": growth.get("epsgrowthTTM"),
        "returnOnEquity": ratios.get("returnOnEquityTTM"),
        "returnOnCapitalEmployed": ratios.get("returnOnCapitalEmployedTTM") or ratios.get("returnOnInvestedCapitalTTM"),
        "debtToEquity": ratios.get("debtEquityRatioTTM"),
        "cik": profile.get("cik")
    }
    return info

# ---------------------------- Endpoint -----------------------------

@app.get("/company/{ticker}", response_model=Snapshot)
def get_company(ticker: str):
    # 1) Try Yahoo first (with retry/backoff)
    use_fmp = False
    try:
        info, t = fetch_yahoo_info_with_retry(ticker)
    except HTTPException as e:
        # If Yahoo rate-limited or failed, switch to FMP
        use_fmp = True

    # 2) If Yahoo failed, use FMP fallback
    if use_fmp:
        info = fetch_fmp_info(ticker)
        t = None  # not used for FMP
        price = info.get("currentPrice")
        name  = info.get("longName") or ticker.upper()
        mcap  = info.get("marketCap")
    else:
        # normal Yahoo path (+fallback for price)
        name  = info.get("longName") or info.get("shortName") or ticker.upper()
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None and t is not None:
            try:
                price = t.fast_info.get("last_price")
            except Exception:
                price = None
        mcap  = info.get("marketCap")

    # 3) Build metrics in a provider-agnostic way
    metrics = dict(
        pe_fwd       = info.get("forwardPE"),
        pe       = info.get("PE"),
        peg_fwd      = info.get("pegRatio"),
        gross_margin = info.get("grossMargins"),
        op_margin    = info.get("operatingMargins"),
        rev_yoy      = info.get("revenueGrowth"),
        eps_yoy      = info.get("earningsGrowth"),
        roic         = info.get("returnOnCapitalEmployed") or info.get("returnOnInvestedCapitalTTM") or info.get("returnOnEquity"),
        de_ratio     = info.get("debtToEquity"),
    )

    # 4) SEC filings link (works with either provider if CIK present)
    filings = {"latest_10q_or_10k": latest_filing(info.get("cik"))}

    # 5) Composite → stars
    composite = score_company(metrics)
    stars, stars_text = stars_from_score(composite)

    # 6) Note which provider answered
    provider_note = "Yahoo primary" if not use_fmp else "FMP fallback (Yahoo blocked/crumb/rate limit)"

    return Snapshot(
        ticker=ticker.upper(),
        name=name,
        price=price,
        market_cap=mcap,
        metrics=metrics,
        composite_score=composite,
        stars=stars,
        stars_text=stars_text,
        filings=filings,
        notes=f"{provider_note}. Data cached 1h."
    )
