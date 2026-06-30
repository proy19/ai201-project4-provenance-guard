"""
Provenance Guard — Flask API
=============================
Endpoints
---------
POST /submit    Classify submitted content
POST /appeal    Lodge a creator appeal

Rate limiting
-------------
Token-bucket limiter, per client IP.
  /submit   10 requests burst, 1 per second steady
  /appeal    5 requests burst, 0.2 per second steady (1 per 5 s)

Run
---
    export GROQ_API_KEY=gsk_...          # optional; stub used if absent
    python app.py                        # default: http://127.0.0.1:5000

    python app.py --host 0.0.0.0 --port 8080 --debug
"""

import argparse
import logging
import uuid
from datetime import datetime, timezone, timedelta

from flask import Flask, request, jsonify, g

import audit_log
import confidence
import labels
import signal1
import signal2
from rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("provenance_guard")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Two separate limiters: submit is heavier, appeals are rarer
_submit_limiter = RateLimiter(capacity=10, refill_rate=1.0)    # 10 burst, 1/s
_appeal_limiter = RateLimiter(capacity=5,  refill_rate=0.2)    # 5 burst, 1/5s


def _client_ip() -> str:
    """Best-effort client IP, respecting X-Forwarded-For if present."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_check(limiter: RateLimiter):
    """Run the limiter for the current request. Aborts with 429 if exceeded."""
    ip = _client_ip()
    allowed, retry_after = limiter.check(ip)
    if not allowed:
        resp = jsonify({
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please wait before retrying.",
            "retry_after_seconds": retry_after,
        })
        resp.status_code = 429
        resp.headers["Retry-After"] = str(int(retry_after) + 1)
        return resp
    return None


# ---------------------------------------------------------------------------
# POST /submit
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["POST"])
def submit():
    # Rate limit
    blocked = _rate_check(_submit_limiter)
    if blocked:
        return blocked

    # Validate input
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_request", "message": "Request body must be JSON."}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "invalid_request", "message": "'text' field is required and must not be empty."}), 400
    if len(text) > 50_000:
        return jsonify({"error": "invalid_request", "message": "'text' must be 50,000 characters or fewer."}), 400

    content_id = data.get("content_id") or str(uuid.uuid4())

    # Signal 1 — Groq semantic classifier
    try:
        s1 = signal1.classify(text)
    except RuntimeError as exc:
        logger.error("Signal 1 failed: %s", exc)
        return jsonify({"error": "upstream_error", "message": "Signal 1 (Groq) unavailable. Please try again."}), 502

    # Signal 2 — Stylometric heuristics
    s2 = signal2.classify(text)

    # Confidence scoring
    scoring = confidence.score(s1, s2)

    # Transparency label
    label_payload = labels.render_full(scoring, content_id)

    # Audit log
    log_id = audit_log.log_submission(
        content_id=content_id,
        signal1=s1,
        signal2=s2,
        scoring=scoring,
        label_text=label_payload["label"],
    )

    logger.info(
        "submit content_id=%s category=%s score=%.3f log_id=%s",
        content_id, scoring.category, scoring.combined_score, log_id,
    )

    return jsonify({
        "content_id": content_id,
        "label": label_payload["label"],
        "short_category": label_payload["short_category"],
        "description": label_payload["description"],
        "score": scoring.combined_score,
        "badge_color": label_payload["badge_color"],
        "contributing_features": scoring.contributing_features,
        "signal_detail": {
            "signal_1": {
                "score": s1["score"],
                "rationale": s1["rationale"],
                "flags": s1["flags"],
            },
            "signal_2": {
                "score": s2["score"],
                "label": s2["label"],
                "features": s2["features"],
            },
        },
        "appeal_url": f"/appeal",
        "log_id": log_id,
    }), 200


# ---------------------------------------------------------------------------
# POST /appeal
# ---------------------------------------------------------------------------

@app.route("/appeal", methods=["POST"])
def appeal():
    # Rate limit
    blocked = _rate_check(_appeal_limiter)
    if blocked:
        return blocked

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_request", "message": "Request body must be JSON."}), 400

    content_id = data.get("content_id", "").strip()
    if not content_id:
        return jsonify({"error": "invalid_request", "message": "'content_id' is required."}), 400

    creator_statement = data.get("creator_statement", "").strip()
    if not creator_statement:
        return jsonify({"error": "invalid_request", "message": "'creator_statement' is required."}), 400
    if len(creator_statement) > 5_000:
        return jsonify({"error": "invalid_request", "message": "'creator_statement' must be 5,000 characters or fewer."}), 400

    # Verify the content_id exists
    existing = audit_log.get_submission(content_id)
    if not existing:
        return jsonify({"error": "not_found", "message": f"No submission found for content_id '{content_id}'."}), 404

    # Status update + audit log
    appeal_id, log_id = audit_log.log_appeal(content_id, creator_statement)

    eta = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()

    logger.info(
        "appeal content_id=%s appeal_id=%s log_id=%s",
        content_id, appeal_id, log_id,
    )

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "pending_review",
        "eta": eta,
        "log_id": log_id,
        "message": "Your appeal has been received and will be reviewed by our team within 3 business days.",
    }), 202


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "provenance-guard"}), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not_found", "message": str(e)}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "method_not_allowed", "message": str(e)}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled exception")
    return jsonify({"error": "internal_error", "message": "An unexpected error occurred."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Provenance Guard API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    audit_log.init_db()
    logger.info("Starting Provenance Guard on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
