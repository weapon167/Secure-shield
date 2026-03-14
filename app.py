 
import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template, request, send_file, redirect
from flask_socketio import SocketIO, emit
import sqlite3
import io
import os
import pandas as pd                  # needed to build the feature DataFrame before prediction
from datetime import datetime
import joblib

from features import extract_features   # our feature extractor that converts a URL into numbers


# =============================================================================
# TRUSTED DOMAIN WHITELIST
# =============================================================================
# Any URL whose hostname ends with one of these suffixes is automatically
# treated as legitimate and skips the ML model entirely.
# This prevents .ac.ke, .edu, .go.ke etc. from ever being flagged as phishing.
# Add more entries here if you find other legitimate domains being flagged.
# =============================================================================
TRUSTED_SUFFIXES = [
    ".ac.ke",       # Kenyan academic institutions e.g. zetech.ac.ke
    ".edu",         # global university domains
    ".edu.ke",      # Kenyan university sub-variant
    ".go.ke",       # Kenyan government websites
    ".gov",         # US and other government websites
    ".gov.ke",      # Kenyan government sub-variant
    ".mil",         # military domains
    ".edu.au",      # Australian universities
    ".ac.uk",       # UK academic institutions
    ".nhs.uk",      # UK national health service
]

# Exact hostnames that should always be treated as legitimate regardless of path
TRUSTED_EXACT = [
    "google.com", "www.google.com",
    "youtube.com", "www.youtube.com",
    "facebook.com", "www.facebook.com",
    "twitter.com", "www.twitter.com",
    "instagram.com", "www.instagram.com",
    "linkedin.com", "www.linkedin.com",
    "microsoft.com", "www.microsoft.com",
    "apple.com", "www.apple.com",
    "amazon.com", "www.amazon.com",
    "wikipedia.org", "www.wikipedia.org",
    "github.com", "www.github.com",
    "zetech.ac.ke", "www.zetech.ac.ke",    # your school - always trusted
    "moodle.zetech.ac.ke",                 # school learning portal
    "portal.zetech.ac.ke",                 # school student portal
]


def extract_hostname(url: str) -> str:
    """
    Pulls just the hostname out of a full URL.
    e.g. "https://portal.zetech.ac.ke/login" → "portal.zetech.ac.ke"
    """
    try:
        host = url.lower().strip()               # lowercase and strip spaces
        host = host.split("://")[-1]             # removes "https://" or "http://"
        host = host.split("/")[0]                # removes everything after the first slash
        host = host.split("?")[0]                # removes query string if no slash before it
        host = host.split(":")[0]                # removes port number e.g. ":8080"
        return host
    except Exception:
        return ""                                # return empty string if anything goes wrong


def is_trusted(url: str) -> bool:
    """
    Returns True if the URL belongs to a known-trusted domain.
    Checks both exact hostname matches and trusted suffix matches.
    """
    hostname = extract_hostname(url)             # get the hostname from the URL

    if hostname in TRUSTED_EXACT:                # check against the exact whitelist first
        return True

    for suffix in TRUSTED_SUFFIXES:              # loop through each trusted suffix
        if hostname.endswith(suffix):            # check if hostname ends with that suffix
            return True

    return False                                 # not in whitelist - proceed to ML model


# =============================================================================
# LOAD MODEL AND FEATURE COLUMN ORDER
# =============================================================================

model = joblib.load("model/phish_model.pkl")                        # loads the trained Random Forest model
feature_columns = joblib.load("model/feature_columns.pkl")          # loads the exact column order used during training


def build_feature_row(url: str) -> pd.DataFrame:
    """
    Converts a single URL into a feature DataFrame that matches exactly
    what the model was trained on - same columns, same order, same derived features.
    """
    _, feat_dict = extract_features(url)         # extract the 11 base features from the URL

    feat_df = pd.DataFrame([feat_dict])           # wrap the dict in a single-row DataFrame

    # Add the same derived features that were added during training
    feat_df["digit_ratio"] = feat_df["digits"] / (feat_df["url_length"] + 1)       # digit density
    feat_df["hyphen_ratio"] = feat_df["hyphens"] / (feat_df["url_length"] + 1)     # hyphen density

    feat_df["suspicion_score"] = (
        feat_df["at_symbol"] * 10 +                                  # @ symbol is a near-certain phishing flag
        feat_df["hyphens"] * 2 +                                     # each hyphen adds moderate suspicion
        (feat_df["url_length"] > 75).astype(int) * 3 +               # unusually long URLs are suspicious
        (feat_df["dots"] > 4).astype(int) * 2 +                      # excessive dots mean excessive subdomains
        feat_df["ip_in_url"] * 8 +                                   # raw IP address is a strong phishing signal
        feat_df["subdomain_count"] * 2                               # each extra subdomain adds suspicion
    )

    feat_df = feat_df[feature_columns]            # reorder columns to exactly match training order

    return feat_df                                # return the aligned feature DataFrame


def predict_url(url: str):
    """
    Main prediction function.
    Returns a risk score from 0 to 100 where:
      0-14   = Very likely legitimate
      15-54  = Suspicious - needs caution
      55-100 = Very likely phishing

    If the URL is on the trusted whitelist, returns 0 immediately
    without running the ML model at all.
    """

    if is_trusted(url):                          # check whitelist first - trusted domains always score 0
        return 0.0

    feat_df = build_feature_row(url)             # convert URL to feature numbers

    # predict_proba returns [[prob_legit, prob_phishing]]
    # we take index [0][1] which is the probability of being phishing (class 1)
    raw_prob = model.predict_proba(feat_df)[0][1]

    # Convert probability (0.0-1.0) to a percentage score (0-100)
    score = raw_prob * 100

    return round(score, 2)                       # round to 2 decimal places for clean display


# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cyber-vault-tech-2026'           # secret key for session security
socketio = SocketIO(app, cors_allowed_origins="*")            # allows real-time socket connections from any origin

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")   # path to the SQLite database file


def get_db():
    """Opens a connection to the SQLite database and returns it."""
    conn = sqlite3.connect(DB_PATH)          # connect to the database file
    conn.row_factory = sqlite3.Row           # makes rows accessible as dicts instead of plain tuples
    return conn


def init_db():
    """Creates the history table if it does not already exist."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                status TEXT,
                score REAL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)                                 # defines table columns: id, url, status, score, timestamp
        db.commit()                          # saves the table creation to disk


init_db()    # run database setup when the app starts


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def home():
    """Loads the homepage and shows the last 10 scanned URLs from history."""
    db = get_db()
    history = db.execute(
        "SELECT url, status, score, date FROM history ORDER BY id DESC LIMIT 10"
    ).fetchall()                             # fetches the 10 most recent scan records
    db.close()
    return render_template("index.html", history=history)   # passes history to the template


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives a URL from the form, runs analysis, and returns results.
    All interface variables (prediction, score, url, reasons, history)
    are passed to the same index.html template - no UI changes.
    """

    url = request.form.get("url", "").strip()   # gets the URL the user typed in the form

    if not url:                                  # if the form was submitted empty, go back to homepage
        return redirect("/")

    # ── Real-time log updates sent to the browser via SocketIO ──────────────
    socketio.emit('log_update', {'msg': '> INITIALIZING SCANNER...'})
    socketio.emit('log_update', {'msg': '> Checking trusted domain whitelist...'})

    trusted = is_trusted(url)                    # check if this URL is on the trusted whitelist

    if trusted:
        socketio.emit('log_update', {'msg': '> Trusted domain detected. Skipping ML scan.'})
    else:
        socketio.emit('log_update', {'msg': '> Extracting structural URL features...'})
        socketio.emit('log_update', {'msg': '> Running Random Forest classifier...'})
        socketio.emit('log_update', {'msg': '> Calibrating probability score...'})

    # ── Run the prediction ───────────────────────────────────────────────────
    score = predict_url(url)                     # get risk score 0-100 from our predict function

    socketio.emit('log_update', {'msg': f'> Analysis complete. Risk Score: {score:.1f}%'})

    # ── Classify based on score thresholds ──────────────────────────────────
    # Threshold is set higher than default (0.50) to reduce false positives on legitimate sites
    if score < 20:                               # below 20% = strong confidence it is legitimate
        status = "Legitimate"
    elif score < 60:                             # 20-59% = some suspicious features but not conclusive
        status = "Suspicious"
    else:                                        # 60% and above = model is confident this is phishing
        status = "Phishing"

    # ── Build explainable reasons list (shown to user as AI explanation) ────
    reasons = []

    if trusted:
        reasons.append("Domain is on the trusted whitelist (academic/government institution).")

    if score >= 60:                              # only show NLP warning if score is actually high
        reasons.append("ML model detected structural patterns common in credential theft URLs.")

    if url.count('.') > 4:                       # more than 4 dots suggests stacked fake subdomains
        reasons.append("Excessive subdomains detected — a common phishing tactic.")

    if "@" in url:                               # @ in URL is almost always malicious
        reasons.append("URL contains '@' symbol used for domain masking.")

    if url.startswith("http://"):                # HTTP (no S) means no encryption
        reasons.append("Connection is unencrypted (HTTP). No SSL/TLS protection.")

    if len(url) > 80:                            # very long URLs hide the real destination at the end
        reasons.append("URL length is unusually long — may be hiding the real domain.")

    if not reasons:                              # if no specific flags were triggered, give a clean message
        reasons.append("No suspicious structural patterns detected.")

    # ── Save scan result to database ─────────────────────────────────────────
    with get_db() as db:
        db.execute(
            "INSERT INTO history (url, status, score) VALUES (?, ?, ?)",
            (url, status, score)
        )                                        # inserts the URL, verdict, and score into the history table
        db.commit()                              # saves the insert to disk

    # ── Fetch updated history to display below the result ────────────────────
    db = get_db()
    history = db.execute(
        "SELECT url, status, score, date FROM history ORDER BY id DESC LIMIT 10"
    ).fetchall()                                 # gets the latest 10 records including the one just saved
    db.close()

    # Returns the same index.html template with all result variables injected
    return render_template(
        "index.html",
        prediction=status,      # "Legitimate", "Suspicious", or "Phishing"
        score=score,            # numeric risk score 0-100
        url=url,                # the URL that was scanned
        reasons=reasons,        # list of explanation strings shown to the user
        history=history         # last 10 scan records for the history panel
    )


@app.route("/download-report")
def download_report():
    """Generates and downloads a plain-text audit report of all scanned URLs."""
    db = get_db()
    rows = db.execute("SELECT * FROM history").fetchall()    # fetches every scan record ever saved
    db.close()

    output = io.StringIO()                                   # creates an in-memory text buffer
    output.write(f"SECURESHIELD AUDIT REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write("=" * 60 + "\n")
    output.write(f"{'DATE':<20} | {'STATUS':<12} | {'SCORE':<6} | {'URL'}\n")
    output.write("-" * 60 + "\n")

    for r in rows:
        output.write(f"{r['date']:<20} | {r['status']:<12} | {r['score']:<6}% | {r['url']}\n")   # writes one line per scan

    mem = io.BytesIO()                                       # creates an in-memory binary buffer for file download
    mem.write(output.getvalue().encode('utf-8'))             # encodes the text as bytes
    mem.seek(0)                                              # rewinds the buffer to the start so Flask can read it

    return send_file(
        mem,
        as_attachment=True,
        download_name="Security_Audit_Report.txt",
        mimetype="text/plain"
    )                                                        # sends the buffer as a downloadable .txt file


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)                 # starts the Flask development server on port 5000