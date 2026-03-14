# =============================================================================
# features.py  -  URL Feature Extractor
# Author : Collins Maseret Juma | Zetech University
# =============================================================================
# PURPOSE:
#   Takes a raw URL string and converts it into 11 numbers.
#   These numbers describe the structural behaviour of the URL.
#   The trained model reads these numbers - not the URL text itself.
# =============================================================================

import pandas as pd   # used to create a DataFrame from the feature dict
import re             # used to search for IP address patterns inside URLs


def extract_features(url: str):
    """
    Extract 11 numerical features from a single URL string.

    Parameters
    ----------
    url : str   The raw URL to analyse e.g. "https://www.google.com/search?q=test"

    Returns
    -------
    df       : pd.DataFrame  single-row DataFrame for ML pipeline input
    features : dict          same values as a plain dict for display/explanations
    """

    url = str(url).strip()   # makes sure input is a string and removes leading/trailing spaces

    # ------------------------------------------------------------------
    # FEATURE 1 - URL LENGTH
    # counts the total number of characters in the URL
    # phishing URLs are often very long to hide the real destination at the end
    # legitimate URLs are usually under 75 characters
    # ------------------------------------------------------------------
    url_length = len(url)

    # ------------------------------------------------------------------
    # FEATURE 2 - DOT COUNT
    # counts how many dots "." appear in the URL
    # legitimate sites usually have 1-2 dots e.g. "google.com" or "mail.google.com"
    # phishing sites stack sub-domains to look official: "secure.paypal.login.verify.evil.com"
    # ------------------------------------------------------------------
    dots = url.count('.')

    # ------------------------------------------------------------------
    # FEATURE 3 - HYPHEN COUNT
    # counts how many hyphens "-" appear in the full URL
    # attackers use hyphens to mimic real brands: "pay-pal-secure-login.com"
    # legitimate corporate domains rarely use hyphens in their main domain
    # ------------------------------------------------------------------
    hyphens = url.count('-')

    # ------------------------------------------------------------------
    # FEATURE 4 - AT SYMBOL PRESENCE  (1 = found, 0 = not found)
    # checks whether the URL contains an "@" character
    # browsers ignore everything BEFORE the @ symbol in a URL
    # attackers exploit this: "https://google.com@evil-site.com" goes to evil-site.com
    # the @ symbol in a URL is almost always a phishing indicator
    # ------------------------------------------------------------------
    at_symbol = 1 if '@' in url else 0

    # ------------------------------------------------------------------
    # FEATURE 5 - DIGIT COUNT
    # counts how many numeric digits 0-9 appear anywhere in the URL
    # legitimate domains are word-based e.g. "amazon.com"
    # phishing URLs often embed raw IP addresses or random numbers: "http://192.168.1.55/bank"
    # ------------------------------------------------------------------
    digits = sum(c.isdigit() for c in url)   # loops over each character and counts those that are digits

    # ------------------------------------------------------------------
    # FEATURE 6 - SLASH COUNT
    # counts how many forward slashes "/" appear in the URL
    # every URL has at least 2 (from "https://")
    # a very deep path with many slashes can indicate redirect chains used by phishing pages
    # ------------------------------------------------------------------
    slashes = url.count('/')

    # ------------------------------------------------------------------
    # FEATURE 7 - HTTPS PRESENCE  (1 = HTTPS, 0 = HTTP only)
    # checks whether the URL starts with "https" (secure) or just "http" (not secure)
    # NOTE: HTTPS alone does NOT mean the site is safe
    # phishing sites also get free SSL certificates, so this is just one signal among many
    # ------------------------------------------------------------------
    https = 1 if url.lower().startswith('https') else 0

    # ------------------------------------------------------------------
    # FEATURE 8 - QUERY STRING PRESENCE  (1 = has "?", 0 = none)
    # checks if the URL has a query string (the part after "?")
    # query strings are normal for search pages and forms
    # combined with other signals, excessive query parameters suggest phishing redirect tricks
    # ------------------------------------------------------------------
    query = 1 if '?' in url else 0

    # ------------------------------------------------------------------
    # FEATURE 9 - IP ADDRESS IN URL  (1 = IP found, 0 = not found)
    # uses a regular expression to search for an IPv4 address pattern (e.g. 192.168.0.1) in the URL
    # legitimate websites always use domain names, never raw IP addresses
    # finding an IP address in the URL is a strong phishing indicator
    # ------------------------------------------------------------------
    ip_pattern = re.compile(r'(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})')   # pattern matches x.x.x.x format
    ip_in_url = 1 if ip_pattern.search(url) else 0                             # 1 if IP found anywhere in the URL

    # ------------------------------------------------------------------
    # FEATURE 10 - URL DEPTH  (number of directory levels in the path)
    # counts how many folders deep the URL goes after the domain
    # example: "https://evil.com/a/b/c/d/e" has depth 5
    # legitimate sites rarely go deeper than 3-4 directory levels
    # phishing pages often bury the fake login form deep in a path to avoid quick inspection
    # ------------------------------------------------------------------
    try:
        path = url.split("://", 1)[-1]    # removes the "https://" or "http://" prefix
        path_part = path.split("?")[0]    # removes the query string (everything after "?")
        depth = path_part.count('/') - 1  # counts remaining slashes; subtract 1 for the domain separator
        url_depth = max(depth, 0)         # makes sure depth never goes below zero
    except Exception:
        url_depth = 0   # if anything goes wrong, default to 0 so training does not crash

    # ------------------------------------------------------------------
    # FEATURE 11 - SUBDOMAIN COUNT
    # counts how many sub-domains appear before the main domain name
    # example: "secure.paypal.login.verify.evil.com" has 4 sub-domains
    # legitimate sites usually have 0 or 1 sub-domain (e.g. "www")
    # excessive sub-domains are used to make phishing URLs look like real bank/payment pages
    # ------------------------------------------------------------------
    try:
        hostname = url.split("://", 1)[-1].split("/")[0]   # isolates just the hostname part of the URL
        hostname = hostname.split(":")[0]                   # removes port number if present e.g. ":8080"
        parts = hostname.split(".")                         # splits hostname into individual labels by dot
        subdomain_count = max(len(parts) - 2, 0)           # sub-domains = total parts minus domain and TLD
    except Exception:
        subdomain_count = 0   # default to 0 if parsing fails

    # ------------------------------------------------------------------
    # ASSEMBLE ALL 11 FEATURES INTO A DICTIONARY
    # ------------------------------------------------------------------
    features = {
        "url_length"      : url_length,       # total character count of the URL
        "dots"            : dots,             # number of dots in the URL
        "hyphens"         : hyphens,          # number of hyphens in the URL
        "at_symbol"       : at_symbol,        # 1 if @ is present, 0 if not
        "digits"          : digits,           # count of numeric digit characters
        "slashes"         : slashes,          # count of forward slashes
        "https"           : https,            # 1 if URL uses HTTPS, 0 if HTTP only
        "query"           : query,            # 1 if URL has a query string, 0 if not
        "ip_in_url"       : ip_in_url,        # 1 if an IP address was found in the URL, 0 if not
        "url_depth"       : url_depth,        # number of directory levels in the URL path
        "subdomain_count" : subdomain_count,  # number of sub-domains before the main domain
    }

    return pd.DataFrame([features]), features   # returns a single-row DataFrame AND the raw dict