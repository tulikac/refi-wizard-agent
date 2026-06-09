import os
import logging
import json
import time
import random
import traceback
from openai import OpenAI
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

# Python 3.14 sandbox image is missing a default CA bundle path — point SSL at certifi
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

app = Flask(__name__)

# ── OpenTelemetry ────────────────────────────────────────────────────────────
#
# Embr injects OTEL_EXPORTER_OTLP_ENDPOINT pointing at the in-cluster OTLP
# collector (gRPC, localhost:4317). The collector then forwards traces+metrics
# to whatever destinations are configured via the project telemetry API
# (see EMBR_TELEMETRY.md). We send everything to that one endpoint and let
# Embr fan out — no hardcoded URLs in the app.

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

_tracer = None
_meter = None
_chat_counter = None
_chat_latency = None
_home_select_counter = None
_home_price_hist = None
_home_bedrooms_hist = None
_chat_interactions = None
_app_errors = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    _otel_available = True
except ImportError:
    _otel_available = False

if _otel_available and OTEL_ENDPOINT:
    try:
        resource = Resource.create({"service.name": "home-finder", "service.version": "1.0.0"})

        # Traces → Embr OTLP collector (gRPC, insecure inside the sandbox)
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True))
        )
        trace.set_tracer_provider(trace_provider)
        _tracer = trace.get_tracer("home-finder")

        # Metrics → same Embr OTLP collector; project telemetry config fans out to Prometheus
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True),
            export_interval_millis=15000,
        )
        metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(metric_provider)
        _meter = metrics.get_meter("home-finder")

        _chat_counter = _meter.create_counter("chat.requests", description="Number of chat requests")
        _chat_latency = _meter.create_histogram("chat.latency_ms", description="Chat response latency in ms")
        _home_select_counter = _meter.create_counter("home.selections", description="Number of home selections")
        _home_price_hist = _meter.create_histogram("home.price", description="Price of selected homes in dollars")
        _home_bedrooms_hist = _meter.create_histogram("home.bedrooms", description="Bedrooms in selected homes")
        _chat_interactions = _meter.create_counter("chat.interactions", description="User-agent message exchanges")
        _app_errors = _meter.create_counter("app.errors", description="Application errors")

        # Auto-instrument Flask and outbound HTTP
        FlaskInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()

        logger.info("OpenTelemetry initialized → OTLP endpoint: %s", OTEL_ENDPOINT)
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed (continuing without): %s", exc)

ENDPOINT = os.environ.get("AZURE_AI_ENDPOINT", "")
API_KEY = os.environ.get("AZURE_AI_API_KEY", "")
AGENT_NAME = os.environ.get("AGENT_NAME", "refi-wizard")
AGENT_VERSION = os.environ.get("AGENT_VERSION", "1")

if not ENDPOINT or not API_KEY:
    logger.warning("AZURE_AI_ENDPOINT / AZURE_AI_API_KEY not set — /chat will fail until configured")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
ERROR_RATE = float(os.environ.get("SYNTHETIC_ERROR_RATE", "0.15"))  # ~15% of requests


# ── Synthetic Errors ─────────────────────────────────────────────────────────

class DatabaseConnectionError(Exception):
    """Simulated database connection failure."""
    pass

class AgentTimeoutError(Exception):
    """Simulated agent response timeout."""
    pass

class RateLimitExceededError(Exception):
    """Simulated API rate limit exceeded."""
    pass

class InvalidHomeDataError(Exception):
    """Simulated corrupt home listing data."""
    pass

_SYNTHETIC_ERRORS = [
    (DatabaseConnectionError, "Connection to PostgreSQL refused: host=db.internal port=5432 - timeout after 5000ms"),
    (AgentTimeoutError, "Agent 'refi-wizard' did not respond within 30s deadline. upstream_latency=30.2s, retries=2"),
    (RateLimitExceededError, "Azure OpenAI rate limit exceeded: 429 Too Many Requests. retry_after=12s, tokens_used=48201/50000"),
    (InvalidHomeDataError, "Home listing id=42 has invalid data: price=None, sqft=-1. Validation failed at models.Home.validate()"),
]


def _maybe_inject_error():
    """Randomly raise a synthetic error to test error telemetry."""
    if random.random() < ERROR_RATE:
        err_cls, msg = random.choice(_SYNTHETIC_ERRORS)
        raise err_cls(msg)

# City coordinates for geomap
CITY_COORDS = {
    "Austin": (30.2672, -97.7431),
    "Denver": (39.7392, -104.9903),
    "Raleigh": (35.7796, -78.6382),
    "Phoenix": (33.4484, -112.0740),
    "Charleston": (32.7765, -79.9311),
    "Portland": (45.5152, -122.6784),
    "Nashville": (36.1627, -86.7816),
    "Tampa": (27.9506, -82.4572),
}

# ── Database ─────────────────────────────────────────────────────────────────

SAMPLE_HOMES = [
    {"address": "742 Evergreen Terrace", "city": "Austin", "state": "TX", "price": 425000, "bedrooms": 3, "bathrooms": 2.0, "sqft": 1850, "description": "Charming single-story home with updated kitchen, hardwood floors, and a spacious backyard with mature trees.", "image_url": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400&h=260&fit=crop"},
    {"address": "1200 Lakeview Dr", "city": "Denver", "state": "CO", "price": 310000, "bedrooms": 2, "bathrooms": 1.0, "sqft": 1200, "description": "Cozy mountain-view condo near downtown with open floor plan and in-unit laundry.", "image_url": "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=400&h=260&fit=crop"},
    {"address": "88 Willow Creek Rd", "city": "Raleigh", "state": "NC", "price": 550000, "bedrooms": 4, "bathrooms": 3.0, "sqft": 2600, "description": "Spacious colonial with a two-car garage, finished basement, and large deck overlooking a wooded lot.", "image_url": "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=400&h=260&fit=crop"},
    {"address": "455 Sunset Blvd", "city": "Phoenix", "state": "AZ", "price": 289000, "bedrooms": 2, "bathrooms": 2.0, "sqft": 1100, "description": "Modern desert retreat with solar panels, quartz countertops, and a community pool.", "image_url": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=400&h=260&fit=crop"},
    {"address": "33 Harbor Walk", "city": "Charleston", "state": "SC", "price": 675000, "bedrooms": 4, "bathrooms": 2.5, "sqft": 2200, "description": "Historic district gem with wraparound porch, exposed brick, and walking distance to waterfront.", "image_url": "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=400&h=260&fit=crop"},
    {"address": "910 Pine Ridge Ct", "city": "Portland", "state": "OR", "price": 485000, "bedrooms": 3, "bathrooms": 2.0, "sqft": 1750, "description": "Craftsman bungalow with original built-ins, updated bathrooms, and a rain garden.", "image_url": "https://images.unsplash.com/photo-1605276374104-dee2a0ed3cd6?w=400&h=260&fit=crop"},
    {"address": "2750 Prairie View Ln", "city": "Nashville", "state": "TN", "price": 365000, "bedrooms": 3, "bathrooms": 2.5, "sqft": 1900, "description": "New construction in growing neighborhood with open concept, smart home features, and energy-efficient appliances.", "image_url": "https://images.unsplash.com/photo-1583608205776-bfd35f0d9f83?w=400&h=260&fit=crop"},
    {"address": "18 Coral Bay Cir", "city": "Tampa", "state": "FL", "price": 520000, "bedrooms": 4, "bathrooms": 3.0, "sqft": 2400, "description": "Waterfront property with private dock, screened lanai, and a resort-style pool.", "image_url": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=400&h=260&fit=crop"},
]


def _get_db():
    """Return a database connection. Uses PostgreSQL if DATABASE_URL is set, else SQLite."""
    if DATABASE_URL:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homes.db")
        return sqlite3.connect(db_path)


def _is_postgres():
    return bool(DATABASE_URL)


def _init_db():
    conn = _get_db()
    cur = conn.cursor()
    if _is_postgres():
        cur.execute("""
            CREATE TABLE IF NOT EXISTS homes (
                id SERIAL PRIMARY KEY,
                address VARCHAR(255), city VARCHAR(100), state VARCHAR(2),
                price INTEGER, bedrooms INTEGER, bathrooms REAL,
                sqft INTEGER, image_url TEXT, description TEXT
            )
        """)
        cur.execute("SELECT COUNT(*) FROM homes")
        count = cur.fetchone()[0]
        if count == 0:
            for i, h in enumerate(SAMPLE_HOMES):
                cur.execute(
                    "INSERT INTO homes (address, city, state, price, bedrooms, bathrooms, sqft, image_url, description) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (h["address"], h["city"], h["state"], h["price"], h["bedrooms"], h["bathrooms"], h["sqft"],
                     h["image_url"], h["description"]),
                )
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS homes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT, city TEXT, state TEXT,
                price INTEGER, bedrooms INTEGER, bathrooms REAL,
                sqft INTEGER, image_url TEXT, description TEXT
            )
        """)
        cur.execute("SELECT COUNT(*) FROM homes")
        count = cur.fetchone()[0]
        if count == 0:
            for i, h in enumerate(SAMPLE_HOMES):
                cur.execute(
                    "INSERT INTO homes (address, city, state, price, bedrooms, bathrooms, sqft, image_url, description) VALUES (?,?,?,?,?,?,?,?,?)",
                    (h["address"], h["city"], h["state"], h["price"], h["bedrooms"], h["bathrooms"], h["sqft"],
                     h["image_url"], h["description"]),
                )
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized with %d homes", max(count, len(SAMPLE_HOMES)))


_init_db()


def _fetch_homes():
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, address, city, state, price, bedrooms, bathrooms, sqft, image_url, description FROM homes ORDER BY id")
    cols = ["id", "address", "city", "state", "price", "bedrooms", "bathrooms", "sqft", "image_url", "description"]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def _fetch_home(home_id):
    conn = _get_db()
    cur = conn.cursor()
    ph = "%s" if _is_postgres() else "?"
    cur.execute(f"SELECT id, address, city, state, price, bedrooms, bathrooms, sqft, image_url, description FROM homes WHERE id = {ph}", (home_id,))
    cols = ["id", "address", "city", "state", "price", "bedrooms", "bathrooms", "sqft", "image_url", "description"]
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(zip(cols, row)) if row else None


def _record_error(exc, context=""):
    """Record an error in OTEL: span exception, error counter, and log."""
    msg = str(exc)[:120]  # truncate for label safety
    if _app_errors:
        _app_errors.add(1, {
            "error.type": type(exc).__name__,
            "error.context": context,
            "error.message": msg,
        })
    if _tracer:
        span = trace.get_current_span()
        if span and span.is_recording():
            span.record_exception(exc)
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            span.set_attribute("error.stacktrace", traceback.format_exc())


# ── OpenAI ───────────────────────────────────────────────────────────────────

def _get_openai_client():
    import httpx
    # NOTE: verify=False — the Embr python 3.14 sandbox lacks a working CA bundle
    # for httpx (even with certifi installed). Traffic is still TLS-encrypted to
    # *.services.ai.azure.com; only chain validation is skipped. Re-enable once
    # the platform image is fixed.
    return OpenAI(
        api_key="placeholder",
        base_url=ENDPOINT.rstrip("/") + "/openai/v1/",
        default_headers={"api-key": API_KEY},
        timeout=55.0,
        http_client=httpx.Client(verify=False, timeout=55.0),
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/homes")
def api_homes():
    try:
        if _tracer:
            with _tracer.start_as_current_span("fetch_homes") as span:
                homes = _fetch_homes()
                span.set_attribute("homes.count", len(homes))
                return jsonify(homes)
        return jsonify(_fetch_homes())
    except Exception as exc:
        _record_error(exc, "api_homes")
        logger.exception("Failed to fetch homes")
        return jsonify({"error": str(exc)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    start = time.time()
    data = request.get_json()
    user_message = data.get("message", "").strip()
    home_id = data.get("home_id")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    home = None
    if home_id:
        home = _fetch_home(home_id)
        if home:
            # Business metrics: selections with geo, price, bedrooms
            lat, lon = CITY_COORDS.get(home["city"], (0, 0))
            if _home_select_counter:
                _home_select_counter.add(1, {
                    "home.city": home["city"], "home.state": home["state"],
                    "home.lat": str(lat), "home.lon": str(lon),
                })
            if _home_price_hist:
                _home_price_hist.record(home["price"], {"home.city": home["city"]})
            if _home_bedrooms_hist:
                _home_bedrooms_hist.record(home["bedrooms"], {"home.city": home["city"]})

            context = (
                f"[Context: The user is looking at this property: {home['address']}, {home['city']} {home['state']} — "
                f"${home['price']:,}, {home['bedrooms']} bed/{home['bathrooms']} bath, {home['sqft']:,} sqft. "
                f"{home['description']}. Help them with financing options for this specific home.]\n\n"
            )
            user_message = context + user_message

    messages = [{"role": "user", "content": user_message}]

    # Track interactions
    if _chat_interactions:
        _chat_interactions.add(1, {"home.id": str(home_id or "none")})

    def _do_chat():
        _maybe_inject_error()
        openai_client = _get_openai_client()
        return openai_client.responses.create(
            input=messages,
            extra_body={
                "agent_reference": {
                    "name": AGENT_NAME,
                    "version": AGENT_VERSION,
                    "type": "agent_reference",
                }
            },
        )

    try:
        if _tracer:
            with _tracer.start_as_current_span("agent_chat") as span:
                span.set_attribute("agent.name", AGENT_NAME)
                span.set_attribute("agent.version", AGENT_VERSION)
                if home:
                    span.set_attribute("home.id", home["id"])
                    span.set_attribute("home.price", home["price"])
                    span.set_attribute("home.city", home["city"])
                response = _do_chat()
                latency_ms = (time.time() - start) * 1000
                span.set_attribute("chat.latency_ms", latency_ms)
                span.set_attribute("chat.response_length", len(response.output_text))
        else:
            response = _do_chat()
            latency_ms = (time.time() - start) * 1000

        if _chat_counter:
            _chat_counter.add(1, {"status": "success"})
        if _chat_latency:
            _chat_latency.record(latency_ms)

        return jsonify({"response": response.output_text})
    except Exception as exc:
        if _chat_counter:
            _chat_counter.add(1, {"status": "error"})
        _record_error(exc, "agent_chat")
        logger.exception("Chat request failed")
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(Exception)
def handle_exception(exc):
    """Global error handler — records all unhandled exceptions in OTEL."""
    _record_error(exc, "unhandled")
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
