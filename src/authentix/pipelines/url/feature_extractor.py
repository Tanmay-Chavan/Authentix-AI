"""URL lexical and metadata feature extraction for phishing detection.

Extracts features from URLs without fetching the page content,
making it safe and fast for real-time analysis.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

import tldextract
import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Feature Names — ordered, fixed-length vector
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Length features (5)
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "fragment_length",

    # Count features (10)
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_signs",
    "num_ampersands",
    "num_equals",
    "num_digits",
    "num_params",
    "num_subdomains",

    # Ratio features (4)
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "vowel_consonant_ratio",

    # Binary features (7)
    "has_https",
    "has_ip_address",
    "has_port",
    "has_at_sign",
    "has_double_slash_redirect",
    "has_hex_encoding",
    "has_punycode",

    # Entropy features (2)
    "domain_entropy",
    "path_entropy",

    # Suspicious pattern features (5)
    "num_suspicious_words",
    "has_tld_in_subdomain",
    "domain_digit_count",
    "longest_word_length",
    "shortest_word_length",
]

NUM_FEATURES = len(FEATURE_NAMES)

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_WORDS = {
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "suspend", "alert", "billing",
    "paypal", "ebay", "amazon", "apple", "microsoft", "google",
    "facebook", "netflix", "support", "helpdesk", "service",
    "wallet", "crypto", "blockchain",
}

# Common TLDs that phishers embed in subdomains
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "uk", "de", "fr",
}


def extract_url_features(url: str) -> dict[str, float]:
    """Extract a fixed-length feature vector from a URL.

    Parameters
    ----------
    url : str
        The URL to analyse.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature names to float values.
        Always returns exactly NUM_FEATURES features.
    """
    url = url.strip()
    parsed = urlparse(url)
    extracted = tldextract.extract(url)

    domain = extracted.top_domain_under_public_suffix or parsed.hostname or ""
    subdomain = extracted.subdomain or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_domain = parsed.hostname or ""

    features: dict[str, float] = {}

    # ── Length features ──
    features["url_length"] = float(len(url))
    features["domain_length"] = float(len(full_domain))
    features["path_length"] = float(len(path))
    features["query_length"] = float(len(query))
    features["fragment_length"] = float(len(fragment))

    # ── Count features ──
    features["num_dots"] = float(url.count("."))
    features["num_hyphens"] = float(url.count("-"))
    features["num_underscores"] = float(url.count("_"))
    features["num_slashes"] = float(url.count("/"))
    features["num_at_signs"] = float(url.count("@"))
    features["num_ampersands"] = float(url.count("&"))
    features["num_equals"] = float(url.count("="))
    features["num_digits"] = float(sum(c.isdigit() for c in url))
    features["num_params"] = float(len(parse_qs(query)))
    features["num_subdomains"] = float(
        len(subdomain.split(".")) if subdomain else 0
    )

    # ── Ratio features ──
    url_len = max(len(url), 1)
    features["digit_ratio"] = sum(c.isdigit() for c in url) / url_len
    features["letter_ratio"] = sum(c.isalpha() for c in url) / url_len
    features["special_char_ratio"] = sum(
        c in string.punctuation for c in url
    ) / url_len

    vowels = sum(c.lower() in "aeiou" for c in url if c.isalpha())
    consonants = sum(c.lower() not in "aeiou" for c in url if c.isalpha())
    features["vowel_consonant_ratio"] = (
        vowels / max(consonants, 1)
    )

    # ── Binary features ──
    features["has_https"] = float(parsed.scheme == "https")
    features["has_ip_address"] = float(
        bool(re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", full_domain))
    )
    features["has_port"] = float(parsed.port is not None)
    features["has_at_sign"] = float("@" in url)
    features["has_double_slash_redirect"] = float(
        "//" in path
    )
    features["has_hex_encoding"] = float(
        bool(re.search(r"%[0-9a-fA-F]{2}", url))
    )
    features["has_punycode"] = float("xn--" in full_domain.lower())

    # ── Entropy features ──
    features["domain_entropy"] = _shannon_entropy(full_domain)
    features["path_entropy"] = _shannon_entropy(path)

    # ── Suspicious pattern features ──
    url_lower = url.lower()
    features["num_suspicious_words"] = float(
        sum(1 for word in SUSPICIOUS_WORDS if word in url_lower)
    )
    features["has_tld_in_subdomain"] = float(
        any(tld in subdomain.lower().split(".") for tld in COMMON_TLDS)
        if subdomain else 0
    )
    features["domain_digit_count"] = float(
        sum(c.isdigit() for c in full_domain)
    )

    # Word-level features from domain
    words = re.split(r"[.\-_/]", full_domain)
    words = [w for w in words if w]
    features["longest_word_length"] = float(
        max((len(w) for w in words), default=0)
    )
    features["shortest_word_length"] = float(
        min((len(w) for w in words), default=0)
    )

    # Sanity check: ensure we always return the right number of features
    assert len(features) == NUM_FEATURES, (
        f"Feature count mismatch: got {len(features)}, expected {NUM_FEATURES}"
    )

    return features


def extract_url_feature_vector(url: str) -> list[float]:
    """Extract features as an ordered list (for ML model input).

    Returns
    -------
    list[float]
        Feature vector with exactly NUM_FEATURES elements,
        in the order defined by FEATURE_NAMES.
    """
    features = extract_url_features(url)
    return [features[name] for name in FEATURE_NAMES]


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string.

    Higher entropy → more random-looking → more suspicious.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
