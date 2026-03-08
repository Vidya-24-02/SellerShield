"""
SellerShield — server.py (Final)

DATA SOURCE PER PLATFORM:
  eBay      → eBay Developer API (real)         ✅
  Etsy      → Etsy Open API (real)              ✅
  Flipkart  → Affiliate API → scrape → estimate ⚡
  Amazon    → Scraping → smart estimate         🔍
  Meesho    → Synthetic ML dataset              🧪
  Myntra    → Synthetic ML dataset              🧪
  Shopsy    → Synthetic ML dataset              🧪
  Snapdeal  → Synthetic ML dataset              🧪
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib, os
import pandas as pd
from database import (save_report, get_report_summary,
                      save_cache, get_cache, save_search, get_stats)
from real_data import fetch_real_data

app = Flask(__name__, static_folder='.')
PLATFORMS = ['Amazon','Flipkart','Meesho','eBay','Etsy','Shopee','Myntra','Snapdeal']

# ── Load ML model ──────────────────────────────────────────────────────
model, feature_names, classes = None, None, None

def load_model():
    global model, feature_names, classes
    if os.path.exists('model/seller_model.pkl'):
        model         = joblib.load('model/seller_model.pkl')
        feature_names = joblib.load('model/feature_names.pkl')
        classes       = joblib.load('model/classes.pkl')
        print(f"✅ ML Model loaded — {len(feature_names)} features")
    else:
        print("⚠️  Model not found — run: python train_model.py")

load_model()

# ── Source badge labels shown in UI ───────────────────────────────────
SOURCE_LABELS = {
    "ebay_api_real":      ("✅ Real Data — eBay API",           "#22c55e"),
    "etsy_api_real":      ("✅ Real Data — Etsy API",           "#22c55e"),
    "flipkart_api_real":  ("✅ Real Data — Flipkart API",       "#22c55e"),
    "flipkart_scrape":    ("🔍 Scraped — Flipkart",             "#a855f7"),
    "amazon_scrape":      ("🔍 Scraped — Amazon",               "#a855f7"),
    "synthetic_ml":       ("🧪 ML Synthetic Estimate",          "#f59e0b"),
    "smart_estimate":     ("📊 Smart Estimate",                 "#f59e0b"),
    "cache":              ("💾 Cached Data",                    "#8888aa"),
}

def source_badge(source):
    label, color = SOURCE_LABELS.get(source, ("📊 Estimate", "#f59e0b"))
    return f'<span style="color:{color};font-weight:700">{label}</span>'

# ── Helpers ────────────────────────────────────────────────────────────

def keyword_risk(name):
    n = name.lower()
    risky = ['free','win','prize','urgent','today','cheap','fake','wholesale','exclusive']
    sus   = ['official','verified','trusted','real','original','genuine']
    score = 25
    for w in risky: score += 12 if w in n else 0
    for w in sus:   score += 6  if w in n else 0
    return min(100, score)

def build_features(raw, platform):
    row = {
        'account_age_months':  float(raw.get('account_age_months', 18)),
        'total_reviews':       float(raw.get('total_reviews', 100)),
        'avg_rating':          float(raw.get('avg_rating', 4.1)),
        'rating_std':          float(raw.get('rating_std', 0.6)),
        'return_rate':         float(raw.get('return_rate', 0.1)),
        'response_time_hrs':   float(raw.get('response_time_hrs', 12)),
        'price_deviation_pct': float(raw.get('price_deviation_pct', 0)),
        'platform_verified':   int(raw.get('platform_verified', 0)),
        'listing_quality':     float(raw.get('listing_quality', 65)),
        'dispute_rate':        float(raw.get('dispute_rate', 0.08)),
        'repeat_buyer_rate':   float(raw.get('repeat_buyer_rate', 0.35)),
        'keyword_risk_score':  float(raw.get('keyword_risk_score', 30)),
    }
    for p in PLATFORMS:
        row[f'platform_{p}'] = int(platform == p)
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

def compute_score(proba):
    idx = {c: i for i, c in enumerate(classes)}
    s = (proba[idx.get('Trusted',0)]       * 92 +
         proba[idx.get('Moderate Risk',1)] * 50 +
         proba[idx.get('High Risk',2)]     * 8)
    return max(5, min(98, round(s)))

def apply_community_penalty(score, rs):
    if rs['count'] == 0:
        return score, None
    penalty   = min(40, rs['count'] * 8)
    new_score = max(5, score - penalty)
    msg = (f"🚨 {rs['count']} user{'s' if rs['count']>1 else ''} reported this seller as fraudulent."
           + (f" Total losses: ₹{rs['total_lost']:,.0f}." if rs['total_lost'] > 0 else ""))
    return new_score, msg

def make_explanation(name, platform, score, label, raw, proba, source, rs, community_msg):
    age     = raw.get('account_age_months', 18)
    reviews = raw.get('total_reviews', 100)
    rstd    = raw.get('rating_std', 0.6)
    verified= raw.get('platform_verified', 0)
    conf    = round(max(proba) * 100, 1)

    age_txt = ("well-established (2+ years)" if age>24 else
               "moderately aged"             if age>8  else "very new (<6 months)")
    rev_txt = ("high review volume"         if reviews>500 else
               "moderate reviews"           if reviews>50  else "suspiciously few reviews")
    rat_txt = ("near-perfect uniform ratings (possible fake reviews)" if rstd<0.2 else
               "consistent ratings"         if rstd<0.6 else "natural rating variation")
    ver_txt = "holds platform verification" if verified else "lacks platform verification"

    if label == 'Trusted':
        verdict = f'Appears <span style="color:#22c55e"><strong>legitimate and trustworthy</strong></span>. No significant fraud indicators.'
    elif label == 'Moderate Risk':
        verdict = f'Shows <span style="color:#f59e0b"><strong>mixed signals</strong></span>. Use secure payment and verify return policy.'
    else:
        verdict = f'Flagged as <span style="color:#ef4444"><strong>HIGH RISK</strong></span>. Multiple fraud indicators. <strong>Avoid this seller.</strong>'

    community_html = (
        f'<br/><br/><div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
        f'border-radius:10px;padding:12px 16px;color:#ef4444;font-weight:600">{community_msg}</div>'
        if community_msg else ''
    )

    return (f'Data Source: {source_badge(source)}<br/><br/>'
            f'Our <strong>Random Forest + Gradient Boosting ensemble</strong> analyzed '
            f'<strong>{name}</strong> on <strong>{platform}</strong> — trust score '
            f'<strong>{score}/100</strong> ({conf}% confidence).<br/><br/>'
            f'The seller has a <strong>{age_txt}</strong> account with <strong>{rev_txt}</strong> '
            f'({int(reviews):,} total). Ratings show <strong>{rat_txt}</strong> (σ={rstd:.2f}). '
            f'Seller {ver_txt}.<br/><br/>'
            f'<strong>Verdict:</strong> {verdict}{community_html}')

def make_flags(raw, platform, rs):
    flags = []
    age      = raw.get('account_age_months', 18)
    reviews  = raw.get('total_reviews', 100)
    rstd     = raw.get('rating_std', 0.6)
    price    = raw.get('price_deviation_pct', 0)
    dispute  = raw.get('dispute_rate', 0.08) * 100
    verified = raw.get('platform_verified', 0)

    if rs['count'] > 0:
        flags.append({'type':'red',
            'title': f'🚨 {rs["count"]} Community Report{"s" if rs["count"]>1 else ""}',
            'desc':  f'Types: {rs["types"] or "fraud"}. Total losses: ₹{rs["total_lost"]:,.0f}'})

    if age < 6:       flags.append({'type':'red',    'title':'🔴 Very New Account',           'desc':f'Only {age:.0f} months old — throwaway account pattern.'})
    elif age > 24:    flags.append({'type':'green',  'title':'✅ Established Account',         'desc':f'Active {age:.0f} months — strong trust signal.'})
    if reviews < 20:  flags.append({'type':'red',    'title':'🔴 Extremely Low Reviews',       'desc':f'Only {int(reviews)} reviews — suspicious for active seller.'})
    elif reviews>500: flags.append({'type':'green',  'title':'✅ High Review Volume',          'desc':f'{int(reviews):,} reviews — genuine activity.'})
    if rstd < 0.15:   flags.append({'type':'red',    'title':'🔴 Uniform Ratings (Fake?)',     'desc':'σ < 0.15 — statistically impossible without manipulation.'})
    elif rstd > 0.5:  flags.append({'type':'green',  'title':'✅ Natural Rating Distribution', 'desc':'Rating variance looks authentic.'})
    if price < -30:   flags.append({'type':'red',    'title':'🔴 Abnormally Low Prices',       'desc':f'{abs(price):.0f}% below market — counterfeit indicator.'})
    if dispute > 30:  flags.append({'type':'red',    'title':'🔴 High Dispute Rate',           'desc':f'{dispute:.0f}% of orders disputed.'})
    if not verified:  flags.append({'type':'yellow', 'title':f'⚠️ Not Verified on {platform}','desc':'No official platform verification badge.'})
    else:             flags.append({'type':'green',  'title':f'✅ {platform} Verified',        'desc':'Passed official vetting process.'})

    return flags[:6]

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def status():
    s = get_stats()
    return jsonify({'model_loaded': model is not None,
                    'features': len(feature_names) if feature_names else 0,
                    'classes': classes or [], **s})

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run: python train_model.py'}), 503

    data        = request.get_json()
    seller_name = data.get('seller_name', '').strip()
    platform    = data.get('platform', 'Amazon')
    if not seller_name:
        return jsonify({'error': 'Seller name required'}), 400

    try:
        # ── Layer 1: Real / synthetic data ────────────────────────────
        cached = get_cache(seller_name, platform)
        if cached:
            raw, source = dict(cached), cached.get('data_source', 'cache')
            print(f"   💾 Cache hit: {seller_name} on {platform}")
        else:
            raw, source = fetch_real_data(seller_name, platform)
            raw['keyword_risk_score'] = keyword_risk(seller_name)
            save_cache(seller_name, platform, raw, source)

        # ── Layer 2: Community reports ─────────────────────────────────
        rs = get_report_summary(seller_name, platform)

        # ── Layer 3: ML prediction ─────────────────────────────────────
        X     = build_features(raw, platform)
        proba = model.predict_proba(X)[0]
        label = model.predict(X)[0]
        score = compute_score(proba)

        # Apply community penalty
        score, community_msg = apply_community_penalty(score, rs)
        if score >= 70:   label = 'Trusted'
        elif score >= 45: label = 'Moderate Risk'
        else:             label = 'High Risk'

        save_search(seller_name, platform, score, label, source)

        return jsonify({
            'score':         score,
            'label':         label,
            'confidence':    round(max(proba) * 100, 1),
            'probabilities': {c: round(float(p)*100,1) for c,p in zip(classes, proba)},
            'data_source':   source,
            'explanation':   make_explanation(seller_name, platform, score, label,
                                              raw, proba, source, rs, community_msg),
            'flags':         make_flags(raw, platform, rs),
            'features': {
                'account_age':     raw.get('account_age_months', 18),
                'total_reviews':   int(raw.get('total_reviews', 100)),
                'avg_rating':      raw.get('avg_rating', 4.1),
                'rating_std':      raw.get('rating_std', 0.6),
                'return_rate':     raw.get('return_rate', 0.1) * 100,
                'response_time':   raw.get('response_time_hrs', 12),
                'listing_quality': raw.get('listing_quality', 65),
                'dispute_rate':    raw.get('dispute_rate', 0.08) * 100,
            },
            'community': rs,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def report():
    data = request.get_json()
    if not all(data.get(k,'').strip() for k in ['name','email','fraud_type','description']):
        return jsonify({'error': 'Missing required fields'}), 400
    save_report(
        seller_name    = data.get('seller','Unknown'),
        platform       = data.get('platform','Unknown'),
        reporter_name  = data.get('name'),
        reporter_email = data.get('email'),
        fraud_type     = data.get('fraud_type'),
        amount_lost    = data.get('amount', 0),
        description    = data.get('description'))
    return jsonify({'success': True,
                    'message': 'Report saved to database. Thank you for protecting the community!'})

@app.route('/api/stats')
def stats_route():
    return jsonify(get_stats())

if __name__ == '__main__':
    print("\n🛡️  SellerShield — Final Version")
    print("━" * 50)
    print(f"   ML Model   : {'✅ Loaded' if model else '❌ Run train_model.py'}")
    print(f"   Database   : ✅ SQLite (sellershield.db)")
    print(f"   eBay       : ✅ Real API (add key in real_data.py)")
    print(f"   Etsy       : ✅ Real API (add key in real_data.py)")
    print(f"   Flipkart   : ⚡ API + scrape fallback")
    print(f"   Amazon     : 🔍 Scrape + smart estimate")
    print(f"   Meesho     : 🧪 Synthetic ML estimate")
    print(f"   Myntra     : 🧪 Synthetic ML estimate")
    print(f"   Shopsy     : 🧪 Synthetic ML estimate")
    print(f"   Snapdeal   : 🧪 Synthetic ML estimate")
    print(f"   URL        : http://localhost:5000")
    print("━" * 50 + "\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
