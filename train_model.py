# =============================================================================
# train_model.py  –  Phishing Detection System Training Script
# Author : Collins Maseret Juma | Zetech University
# =============================================================================

import os                          # lets us create folders (model/) if they don't exist
import pandas as pd                # used to load the CSV and manipulate data in tables
import numpy as np                 # used for numeric operations on arrays
import joblib                      # used to save and load the trained model to/from disk

from sklearn.ensemble import RandomForestClassifier        # the main ML algorithm we use
from sklearn.calibration import CalibratedClassifierCV    # makes probability scores realistic
from sklearn.model_selection import train_test_split      # splits data into train and test sets
from sklearn.metrics import (
    accuracy_score,        # measures % of correct predictions overall
    precision_score,       # of all URLs flagged phishing, how many truly were?
    recall_score,          # of all actual phishing URLs, how many did we catch?
    f1_score,              # balance between precision and recall (single score)
    roc_auc_score,         # measures how well model separates the two classes (0.5-1.0)
    confusion_matrix,      # table showing correct vs wrong predictions broken down by class
    classification_report  # prints precision, recall, f1 per class in a readable format
)
from sklearn.preprocessing import StandardScaler  # scales feature values to same range
from sklearn.pipeline import Pipeline             # chains scaler + model into one object

from features import extract_features             # our custom URL feature extractor


# =============================================================================
# STEP 1 - LOAD AND CLEAN THE DATASET
# =============================================================================

print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

data = pd.read_csv("data/phishing_site_urls.csv")   # reads the CSV file into a table

data["URL"] = data["URL"].astype(str).str.strip()   # forces every URL to be a string and removes spaces

data = data.dropna(subset=["URL", "Label"])          # removes rows where URL or Label is empty/missing
data = data[data["URL"] != ""]                       # removes rows where URL is literally an empty string

data["Label"] = data["Label"].str.lower().str.strip()        # converts labels to lowercase and trims spaces
data = data[data["Label"].isin(["bad", "good"])]             # keeps only rows with valid labels (bad or good)
data["Label"] = data["Label"].map({"bad": 1, "good": 0})     # converts text labels to numbers: bad=1, good=0

print(f"  Total rows after cleaning : {len(data)}")
print(f"  Phishing  (bad)  : {data['Label'].value_counts()[1]}")
print(f"  Legitimate (good): {data['Label'].value_counts()[0]}")


# =============================================================================
# STEP 2 - BALANCE THE DATASET
# =============================================================================
# If we train on 8000 phishing and 2000 legitimate, the model learns to just
# guess "phishing" all the time and still be 80% accurate - that is cheating.
# Equal class sizes force the model to actually learn the difference.

print("\n" + "=" * 60)
print("STEP 2: Balancing dataset...")
print("=" * 60)

bad_urls  = data[data["Label"] == 1]    # separates all phishing rows into their own table
good_urls = data[data["Label"] == 0]    # separates all legitimate rows into their own table

min_class_size = min(len(bad_urls), len(good_urls))   # finds the smaller class so we never over-sample

bad_sample  = bad_urls.sample(n=min_class_size, random_state=42)    # randomly picks N phishing rows
good_sample = good_urls.sample(n=min_class_size, random_state=42)   # randomly picks N legitimate rows

balanced_data = pd.concat([bad_sample, good_sample])                 # stacks the two samples into one table
balanced_data = balanced_data.sample(frac=1, random_state=42)        # shuffles rows so phishing and legit are mixed
balanced_data = balanced_data.reset_index(drop=True)                 # resets row numbers after shuffling

print(f"  Using {min_class_size} samples per class (total: {min_class_size * 2} rows)")
print(f"  Balanced dataset size: {len(balanced_data)}")


# =============================================================================
# STEP 3 - EXTRACT FEATURES FROM EVERY URL
# =============================================================================
# The model cannot read raw text like "https://google.com".
# extract_features() converts each URL into numbers the model can learn from.

print("\n" + "=" * 60)
print("STEP 3: Extracting features from all URLs...")
print("=" * 60)

def safe_extract(url):
    """
    Calls extract_features() safely.
    If a URL is malformed and causes an error, returns zeros instead of crashing.
    """
    try:
        _, feat_dict = extract_features(url)   # calls features.py - returns (DataFrame, dict)
        return feat_dict                        # we only need the dict here
    except Exception:
        return {                                # if URL breaks the extractor, return all zeros
            "url_length": 0, "dots": 0, "hyphens": 0,
            "at_symbol": 0, "digits": 0, "slashes": 0,
            "https": 0, "query": 0, "ip_in_url": 0,
            "url_depth": 0, "subdomain_count": 0
        }

feature_rows = balanced_data["URL"].apply(safe_extract)   # runs safe_extract on every URL row

X = pd.DataFrame(list(feature_rows))   # converts the list of dicts into a proper numeric table

y = balanced_data["Label"].values       # extracts the labels (0 or 1) as a plain array

print(f"  Feature matrix shape: {X.shape}  (rows x features)")
print(f"  Features used: {list(X.columns)}")


# =============================================================================
# STEP 4 - ADD DERIVED (ENGINEERED) FEATURES
# =============================================================================
# These are calculated FROM the base features.
# They capture combined patterns that single counts miss.

print("\n" + "=" * 60)
print("STEP 4: Engineering additional features...")
print("=" * 60)

X["digit_ratio"] = X["digits"] / (X["url_length"] + 1)    # divides digit count by URL length (+1 avoids divide-by-zero)
X["hyphen_ratio"] = X["hyphens"] / (X["url_length"] + 1)  # divides hyphen count by URL length

X["suspicion_score"] = (
    X["at_symbol"] * 10 +                              # @ symbol scores 10 - it is almost never in a real URL
    X["hyphens"] * 2 +                                 # each hyphen adds 2 - phishing domains use many hyphens
    (X["url_length"] > 75).astype(int) * 3 +           # adds 3 if URL is longer than 75 characters
    (X["dots"] > 4).astype(int) * 2 +                  # adds 2 if more than 4 dots (excessive sub-domains)
    X["ip_in_url"] * 8 +                               # adds 8 if a raw IP address is in the URL
    X["subdomain_count"] * 2                            # each extra sub-domain adds 2 to the score
)

print(f"  Extended feature matrix shape: {X.shape}")   # confirms new columns were added


# =============================================================================
# STEP 5 - TRAIN / TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Splitting into train / test sets (80% / 20%)...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,               # X = features table, y = labels array
    test_size=0.20,     # holds back 20% of rows for testing - the model never sees these during training
    random_state=42,    # fixed seed so the split is identical every time you run this
    stratify=y          # ensures both train and test sets have equal proportions of phishing and legitimate
)

print(f"  Training samples : {len(X_train)}")   # number of rows used for training
print(f"  Test samples     : {len(X_test)}")    # number of rows used for evaluation


# =============================================================================
# STEP 6 - BUILD THE MODEL PIPELINE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Building and calibrating the model pipeline...")
print("=" * 60)

base_pipeline = Pipeline([           # Pipeline runs these two steps in sequence automatically
    (
        "scaler",
        StandardScaler()             # scales all feature numbers to a similar range so no single feature dominates
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=50,        # builds 300 separate decision trees and averages their votes
            max_depth=10,            # each tree can go at most 20 levels deep - prevents memorising noise
            min_samples_leaf=4,      # a leaf node needs at least 4 training examples - reduces overfit
            max_features="sqrt",     # each tree only sees sqrt(n_features) columns - adds variety between trees
            class_weight="balanced", # automatically gives more weight to whichever class has fewer samples
            random_state=42,         # fixed seed for reproducible results every run
            n_jobs=-1                # uses all available CPU cores to train faster
        )
    )
])

calibrated_model = CalibratedClassifierCV(
    base_pipeline,       # wraps our pipeline so probability scores are corrected to be realistic
    method="isotonic",   # isotonic regression calibration - more accurate than sigmoid for Random Forests
    cv=3                 # uses 5-fold cross-validation internally to fit the calibration curve
)

print("  Pipeline ready: StandardScaler -> RandomForest (300 trees) -> Isotonic Calibration")


# =============================================================================
# STEP 7 - TRAIN THE MODEL
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Training the model (this may take 1-3 minutes)...")
print("=" * 60)

calibrated_model.fit(X_train, y_train)   # fits the entire pipeline on the training data - this is where learning happens

print("  Training complete.")


# =============================================================================
# STEP 8 - EVALUATE ON THE TEST SET
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Evaluating on the unseen test set...")
print("=" * 60)

y_pred = calibrated_model.predict(X_test)              # predicts 0 or 1 for each test URL
y_prob = calibrated_model.predict_proba(X_test)[:, 1]  # gets the phishing probability (0.0-1.0) for each test URL

accuracy  = accuracy_score(y_test, y_pred)             # overall percentage of correct predictions
precision = precision_score(y_test, y_pred)            # of all URLs we flagged as phishing, % that actually were
recall    = recall_score(y_test, y_pred)               # of all real phishing URLs, % that we successfully caught
f1        = f1_score(y_test, y_pred)                   # single score that balances precision and recall
roc_auc   = roc_auc_score(y_test, y_prob)              # overall ability to separate phishing from legit (1.0 = perfect)
cm        = confusion_matrix(y_test, y_pred)           # 2x2 breakdown: TN, FP, FN, TP

print(f"\n  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  (when we flag phishing, we are right this % of the time)")
print(f"  Recall    : {recall:.4f}  (we caught this fraction of ALL real phishing URLs)")
print(f"  F1-Score  : {f1:.4f}  (balance between precision and recall)")
print(f"  ROC-AUC   : {roc_auc:.4f}  (1.0 = perfect, 0.5 = random guessing)")

print("\n  Confusion Matrix:")
print(f"                    Predicted Legit    Predicted Phishing")
print(f"  Actual Legit  :        {cm[0][0]:<12}       {cm[0][1]}")   # TN = correctly called legit | FP = wrongly flagged
print(f"  Actual Phish  :        {cm[1][0]:<12}       {cm[1][1]}")   # FN = missed phishing        | TP = correctly caught

print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

fp_rate = cm[0][1] / (cm[0][0] + cm[0][1])   # false positive rate = (wrongly flagged legit) divided by (all actual legit)

if fp_rate > 0.05:   # warn if more than 5% of legitimate URLs are being flagged incorrectly
    print(f"  WARNING: False positive rate is {fp_rate:.2%} - model is flagging too many legitimate URLs.")
    print("    Fix: raise min_samples_leaf to 6 or increase the prediction threshold above 0.50.")
else:
    print(f"  OK: False positive rate is {fp_rate:.2%} - acceptable.")


# =============================================================================
# STEP 9 - SAMPLE PROBABILITY PREDICTIONS
# =============================================================================
# Prints known URLs so you can visually verify the model is working correctly
# before saving it. A score near 0% = very likely legit, near 100% = phishing.

print("\n" + "=" * 60)
print("STEP 9: Sample probability predictions...")
print("=" * 60)

sample_urls = [
    ("https://www.google.com",                                       "Legitimate"),
    ("https://www.amazon.com/products",                              "Legitimate"),
    ("https://www.facebook.com/login",                               "Legitimate"),
    ("http://paypal-secure-login.xyz/verify@user",                   "Phishing"),
    ("http://192.168.1.1/bank-login?user=verify&token=abc123",       "Phishing"),
    ("http://free-iphone-winner.com/claim-now-click-here-urgent",    "Phishing"),
]

print(f"\n  {'URL':<55} {'Expected':<12} {'Score':>6}  {'Verdict'}")
print("  " + "-" * 90)

for url, expected in sample_urls:    # loops through each test URL one by one
    try:
        _, feat_dict = extract_features(url)         # extracts the 11 base features for this URL
        feat_df = pd.DataFrame([feat_dict])           # wraps them in a single-row DataFrame

        # Add the exact same derived features we added during training - order must match
        feat_df["digit_ratio"]     = feat_df["digits"] / (feat_df["url_length"] + 1)
        feat_df["hyphen_ratio"]    = feat_df["hyphens"] / (feat_df["url_length"] + 1)
        feat_df["suspicion_score"] = (
            feat_df["at_symbol"] * 10 +
            feat_df["hyphens"] * 2 +
            (feat_df["url_length"] > 75).astype(int) * 3 +
            (feat_df["dots"] > 4).astype(int) * 2 +
            feat_df["ip_in_url"] * 8 +
            feat_df["subdomain_count"] * 2
        )

        prob  = calibrated_model.predict_proba(feat_df)[0][1]        # index [1] = probability it is phishing
        label = "PHISHING" if prob >= 0.50 else "Legitimate"          # 0.50 threshold: above = phishing, below = legit

        display_url = (url[:52] + "...") if len(url) > 55 else url   # trims long URLs so table stays readable
        match_icon  = "OK" if label.lower() == expected.lower() else "WRONG"  # shows if prediction matched expected
        print(f"  {display_url:<55} {expected:<12} {prob:>5.1%}   {label}  {match_icon}")

    except Exception as e:
        print(f"  Could not process: {url} - {e}")   # prints error message but continues to next URL


# =============================================================================
# STEP 10 - SAVE MODEL AND FEATURE COLUMN LIST
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Saving model to disk...")
print("=" * 60)

os.makedirs("model", exist_ok=True)   # creates the model/ folder if it does not already exist

joblib.dump(calibrated_model, "model/phish_model.pkl")    # saves the entire trained pipeline to a .pkl file

feature_columns = list(X.columns)                          # records the exact feature names and order used in training
joblib.dump(feature_columns, "model/feature_columns.pkl")  # saves column list so prediction code can align features correctly

print(f"  Model saved          : model/phish_model.pkl")
print(f"  Feature columns saved: model/feature_columns.pkl")
print(f"  Feature order        : {feature_columns}")
print("\n  Done. Model is ready to use in your Flask app.")
print("=" * 60)