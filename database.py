"""
SellerShield — database.py
SQLite database for:
  1. Storing community fraud reports
  2. Caching real API/scrape data
  3. Search history + stats
"""

import sqlite3
from datetime import datetime

DB_PATH = 'sellershield.db'

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Fraud reports from users
    c.execute('''CREATE TABLE IF NOT EXISTS fraud_reports (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_name    TEXT NOT NULL,
        platform       TEXT NOT NULL,
        reporter_name  TEXT,
        reporter_email TEXT,
        fraud_type     TEXT,
        amount_lost    REAL DEFAULT 0,
        description    TEXT,
        reported_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # Cache real fetched seller data
    c.execute('''CREATE TABLE IF NOT EXISTS seller_cache (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_name         TEXT NOT NULL,
        platform            TEXT NOT NULL,
        account_age_months  REAL,
        total_reviews       INTEGER,
        avg_rating          REAL,
        rating_std          REAL,
        return_rate         REAL,
        response_time_hrs   REAL,
        price_deviation_pct REAL,
        platform_verified   INTEGER,
        listing_quality     REAL,
        dispute_rate        REAL,
        repeat_buyer_rate   REAL,
        data_source         TEXT,
        fetched_at          TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(seller_name, platform)
    )''')

    # Search history
    c.execute('''CREATE TABLE IF NOT EXISTS search_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        seller_name TEXT,
        platform    TEXT,
        trust_score INTEGER,
        label       TEXT,
        data_source TEXT,
        searched_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()

# ── REPORTS ────────────────────────────────────────────────────────────

def save_report(seller_name, platform, reporter_name, reporter_email,
                fraud_type, amount_lost, description):
    conn = get_conn()
    conn.execute('''INSERT INTO fraud_reports
        (seller_name, platform, reporter_name, reporter_email, fraud_type, amount_lost, description)
        VALUES (?,?,?,?,?,?,?)''',
        (seller_name, platform, reporter_name, reporter_email,
         fraud_type, amount_lost or 0, description))
    conn.commit()
    conn.close()

def get_report_summary(seller_name, platform):
    conn = get_conn()
    row = conn.execute('''
        SELECT COUNT(*) as count, SUM(amount_lost) as total_lost,
               GROUP_CONCAT(DISTINCT fraud_type) as types
        FROM fraud_reports
        WHERE LOWER(seller_name)=LOWER(?) AND LOWER(platform)=LOWER(?)
    ''', (seller_name, platform)).fetchone()
    conn.close()
    return {'count': row['count'] or 0,
            'total_lost': row['total_lost'] or 0,
            'types': row['types'] or ''}

def get_recent_reports(limit=10):
    conn = get_conn()
    rows = conn.execute('''SELECT seller_name, platform, fraud_type, amount_lost, reported_at
        FROM fraud_reports ORDER BY reported_at DESC LIMIT ?''', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── CACHE ──────────────────────────────────────────────────────────────

def save_cache(seller_name, platform, features, source):
    conn = get_conn()
    conn.execute('''INSERT OR REPLACE INTO seller_cache
        (seller_name, platform, account_age_months, total_reviews, avg_rating,
         rating_std, return_rate, response_time_hrs, price_deviation_pct,
         platform_verified, listing_quality, dispute_rate, repeat_buyer_rate,
         data_source, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        seller_name, platform,
        features.get('account_age_months'), features.get('total_reviews'),
        features.get('avg_rating'),         features.get('rating_std'),
        features.get('return_rate'),        features.get('response_time_hrs'),
        features.get('price_deviation_pct'),features.get('platform_verified'),
        features.get('listing_quality'),    features.get('dispute_rate'),
        features.get('repeat_buyer_rate'),  source,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def get_cache(seller_name, platform):
    conn = get_conn()
    row = conn.execute('''SELECT * FROM seller_cache
        WHERE LOWER(seller_name)=LOWER(?) AND LOWER(platform)=LOWER(?)''',
        (seller_name, platform)).fetchone()
    conn.close()
    return dict(row) if row else None

# ── HISTORY + STATS ────────────────────────────────────────────────────

def save_search(seller_name, platform, score, label, source):
    conn = get_conn()
    conn.execute('''INSERT INTO search_history
        (seller_name, platform, trust_score, label, data_source)
        VALUES (?,?,?,?,?)''', (seller_name, platform, score, label, source))
    conn.commit()
    conn.close()

def get_stats():
    conn = get_conn()
    total   = conn.execute('SELECT COUNT(*) FROM search_history').fetchone()[0]
    flagged = conn.execute("SELECT COUNT(*) FROM search_history WHERE label='High Risk'").fetchone()[0]
    reports = conn.execute('SELECT COUNT(*) FROM fraud_reports').fetchone()[0]
    conn.close()
    return {'total_analyzed': total, 'total_flagged': flagged, 'total_reports': reports}

init_db()
