"""
Phase 9 — News Sentiment Engine.

Fetches latest market news headlines via yfinance and applies
VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment
analysis to produce a composite Market Sentiment Score.

Functions
---------
fetch_market_news : Retrieve recent news articles for a given ticker.
analyse_sentiment : Score a list of headlines using VADER.
get_market_pulse  : High-level wrapper returning scored headlines + gauge.
"""

import datetime
from typing import Dict, List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


# ---------------------------------------------------------------------------
# News Fetching
# ---------------------------------------------------------------------------

def fetch_market_news(
    ticker: str = "^NSEI",
    max_articles: int = 20,
) -> List[Dict]:
    """Fetch the latest news headlines for a given ticker via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (default ``'^NSEI'`` for Nifty 50).
    max_articles : int
        Maximum number of articles to return.

    Returns
    -------
    list of dict
        Each dict contains ``'title'``, ``'publisher'``, ``'link'``,
        and ``'published'`` keys.
    """
    if yf is None:
        print("   yfinance not installed. Cannot fetch news.")
        return []

    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
    except Exception as exc:
        print(f"   Failed to fetch news for {ticker}: {exc}")
        return []

    articles = []
    for item in news_items[:max_articles]:
        # yfinance >= 0.2.40 returns dicts with varying keys
        title = item.get("title", item.get("content", {}).get("title", ""))
        publisher = item.get("publisher", item.get("content", {}).get("provider", {}).get("displayName", "Unknown"))
        link = item.get("link", item.get("content", {}).get("canonicalUrl", {}).get("url", ""))
        pub_date = item.get("providerPublishTime", None)

        if pub_date and isinstance(pub_date, (int, float)):
            pub_date = datetime.datetime.fromtimestamp(pub_date).strftime(
                "%Y-%m-%d %H:%M"
            )
        elif pub_date is None:
            pub_date = "N/A"

        if title:
            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": pub_date,
            })

    return articles


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

def analyse_sentiment(articles: List[Dict]) -> pd.DataFrame:
    """Apply VADER sentiment scoring to a list of news articles.

    Each headline is scored on a scale of –1 (extremely negative) to
    +1 (extremely positive) using the VADER compound score.

    Parameters
    ----------
    articles : list of dict
        Output from :func:`fetch_market_news`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``title``, ``publisher``, ``published``,
        ``link``, ``compound``, and ``label`` ('Positive' / 'Negative' /
        'Neutral').
    """
    if SentimentIntensityAnalyzer is None:
        print("   vaderSentiment not installed. Cannot score headlines.")
        return pd.DataFrame()

    if not articles:
        return pd.DataFrame()

    analyser = SentimentIntensityAnalyzer()
    rows = []

    for art in articles:
        scores = analyser.polarity_scores(art["title"])
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        rows.append({
            "title": art["title"],
            "publisher": art["publisher"],
            "published": art["published"],
            "link": art["link"],
            "compound": compound,
            "label": label,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-Level Convenience
# ---------------------------------------------------------------------------

def get_market_pulse(
    tickers: Optional[List[str]] = None,
    max_articles_per_ticker: int = 15,
) -> Dict:
    """Return a scored news DataFrame and an overall sentiment gauge value.

    Aggregates news from one or more tickers and computes the mean
    compound score, which serves as the Market Sentiment Score.

    Parameters
    ----------
    tickers : list of str, optional
        Ticker symbols to scan (default: Nifty 50, Bank Nifty, Sensex).
    max_articles_per_ticker : int
        Max headlines per ticker.

    Returns
    -------
    dict
        ``'df'`` — pandas DataFrame of scored headlines.
        ``'score'`` — float in [–1, 1] representing aggregate sentiment.
        ``'label'`` — str overall sentiment label.
    """
    if tickers is None:
        tickers = ["^NSEI", "^NSEBANK", "^BSESN"]

    all_articles: List[Dict] = []
    for tkr in tickers:
        all_articles.extend(
            fetch_market_news(tkr, max_articles=max_articles_per_ticker)
        )

    # De-duplicate by title
    seen_titles: set = set()
    unique_articles: List[Dict] = []
    for art in all_articles:
        if art["title"] not in seen_titles:
            seen_titles.add(art["title"])
            unique_articles.append(art)

    scored_df = analyse_sentiment(unique_articles)

    if scored_df.empty:
        return {"df": scored_df, "score": 0.0, "label": "Neutral"}

    avg_score = scored_df["compound"].mean()

    if avg_score >= 0.05:
        overall_label = "Bullish"
    elif avg_score <= -0.05:
        overall_label = "Bearish"
    else:
        overall_label = "Neutral"

    return {"df": scored_df, "score": avg_score, "label": overall_label}


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  NEWS SENTIMENT ENGINE")
    print("=" * 60)

    pulse = get_market_pulse()
    df = pulse["df"]

    if df.empty:
        print("  No news retrieved. Check your internet connection.")
    else:
        print(f"\n  Headlines Scored: {len(df)}")
        print(f"  Market Sentiment Score: {pulse['score']:.3f} ({pulse['label']})")
        print(f"\n  Top 5 Headlines:")
        for _, row in df.head(5).iterrows():
            sentiment_label = row["label"].upper()
            print(f"    {sentiment_label:8s} [{row['compound']:+.3f}] {row['title'][:80]}")
    print("=" * 60)
