"""
AEGIS AI — Malicious URL Detection
Uses: VirusTotal API (70+ engines) + local heuristics fallback
"""

import re
import base64
import requests

# ── Apni API key yahan paste karo ──────────────────
VT_API_KEY = "08e8b3903f9cd925b44696e49f0f4efc52209b9e345e56b0dad1372dce7b0fa6"

# ── Local heuristic patterns (API ke bina bhi kaam kare) ───
_SUSPICIOUS = [
    r"bit\.ly", r"tinyurl", r"goo\.gl",
    r"\.tk$", r"\.ml$", r"\.ga$", r"\.cf$",
    r"login.*paypal", r"secure.*bank", r"verify.*account",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",   # raw IP
    r"@",                                        # @ trick
    r"free.*prize", r"click.*here.*win",
    r"password.*reset.*urgent",
]

def _heuristic_check(url: str) -> dict | None:
    """Quick local check — returns result if suspicious pattern found."""
    url_lower = url.lower()
    hits = sum(1 for p in _SUSPICIOUS if re.search(p, url_lower))
    if hits > 0:
        confidence = round(min(60.0 + hits * 10.0, 97.0), 2)
        return {"result": "Dangerous", "confidence": confidence}
    return None

def _virustotal_check(url: str) -> dict:
    """Full VirusTotal API scan."""
    try:
        # URL ko base64 encode karo (VT ka requirement)
        url_id  = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        headers = {"x-apikey": VT_API_KEY}

        resp = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers,
            timeout=10
        )

        if resp.status_code == 404:
            # URL pehle scan nahi hui — submit karo
            submit = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url},
                timeout=10
            )
            if submit.status_code == 200:
                return {"result": "Scanning", "confidence": 50.0}
            return {"result": "Safe", "confidence": 70.0}

        if resp.status_code != 200:
            raise Exception(f"VT API error: {resp.status_code}")

        data      = resp.json()
        stats     = data["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0)
        suspicious= stats.get("suspicious", 0)
        total     = sum(stats.values()) or 1

        threat_count = malicious + suspicious

        if threat_count > 0:
            confidence = round(min((threat_count / total) * 100 * 3, 99.0), 2)
            return {"result": "Dangerous", "confidence": confidence}
        else:
            clean      = stats.get("undetected", 0) + stats.get("harmless", 0)
            confidence = round(min((clean / total) * 100, 99.0), 2)
            return {"result": "Safe", "confidence": confidence}

    except requests.exceptions.Timeout:
        return {"result": "Timeout", "confidence": 0.0}
    except Exception as e:
        print(f"VT API Error: {e}")
        return None


def detect_malicious_link(url: str) -> dict:
    """
    Main detection function.
    1. Pehle local heuristics check
    2. Phir VirusTotal API
    3. Fallback to safe if API fails
    """
    # Validate URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Step 1: Local heuristic
    heuristic = _heuristic_check(url)
    if heuristic:
        return heuristic

    # Step 2: VirusTotal
    if VT_API_KEY != "08e8b3903f9cd925b44696e49f0f4efc52209b9e345e56b0dad1372dce7b0fa6":
        vt_result = _virustotal_check(url)
        if vt_result:
            return vt_result

    # Step 3: Fallback
    return {"result": "Safe", "confidence": 85.0}