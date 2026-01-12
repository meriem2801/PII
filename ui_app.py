import os
# =========================
# fix pour question de cache HF en cloud
# =========================
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/home/appuser/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/appuser/.cache/huggingface"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/home/appuser/.cache/sentence_transformers"

import re
import time
import types
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
import googlemaps
from dotenv import load_dotenv
from agents.dispatcher import Dispatcher

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Assistant Mobilité Urbaine",
    page_icon="🧭",
    layout="wide",
)

# =========================
# Load .env (local) + secrets (cloud)
# =========================
load_dotenv()  # safe en cloud, utile en local

def get_secret(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

openai_key = get_secret("OPENAI_API_KEY")
gmaps_key  = get_secret("GOOGLE_MAPS_API_KEY")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if gmaps_key:
    os.environ["GOOGLE_MAPS_API_KEY"] = gmaps_key

gmaps = googlemaps.Client(key=gmaps_key) if gmaps_key else None

# =========================
# Styles (simple + chat clean)
# =========================
st.markdown(
    """
<style>
.main .block-container{ padding-top: 1.2rem; }
.chat-title{ font-size: 26px; font-weight: 700; margin: 0 0 2px 0; }
.chat-subtitle{ color: rgba(0,0,0,0.6); margin: 0 0 14px 0; }
.small-muted{ color: rgba(0,0,0,0.55); font-size: 12px; }
hr{ margin: .8rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# State init
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "user_city" not in st.session_state:
    st.session_state.user_city = None

# --- états pour polling géoloc fiable ---
if "geo_pending" not in st.session_state:
    st.session_state.geo_pending = False
if "geo_started_at" not in st.session_state:
    st.session_state.geo_started_at = 0.0
if "geo_attempt" not in st.session_state:
    st.session_state.geo_attempt = 0
if "geo_payload" not in st.session_state:
    st.session_state.geo_payload = None

if "ip_pending" not in st.session_state:
    st.session_state.ip_pending = False
if "ip_started_at" not in st.session_state:
    st.session_state.ip_started_at = 0.0
if "ip_attempt" not in st.session_state:
    st.session_state.ip_attempt = 0
if "ip_payload" not in st.session_state:
    st.session_state.ip_payload = None

# =========================
# Dispatcher + context
# =========================
@st.cache_resource
def get_dispatcher():
    disp = Dispatcher()
    disp.context = types.SimpleNamespace(location=None, geo_permission=False, city=None)
    return disp

disp = get_dispatcher()
ctx  = disp.context

# =========================
# Geolocation helpers
# =========================
@st.cache_data(ttl=3600)
def reverse_city_osm(lat: float, lon: float) -> str | None:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 10},
            headers={"User-Agent": "mobility-app"},
            timeout=10,
        )
        addr = r.json().get("address", {})
        return addr.get("city") or addr.get("town") or addr.get("village")
    except Exception:
        return None

@st.cache_data(ttl=3600)
def reverse_city_google(lat: float, lon: float) -> str | None:
    if not gmaps:
        return None
    try:
        rev = gmaps.reverse_geocode((lat, lon))
        if not rev:
            return None
        for comp in rev[0].get("address_components", []):
            if "locality" in comp.get("types", []):
                return comp.get("long_name")
        return None
    except Exception:
        return None

def set_city(city: str | None):
    if city:
        st.session_state.user_city = city
        ctx.location = city
        ctx.city = city

# =========================
# ✅ Composants navigateur (GPS + IP) + polling
# =========================
def _browser_geo_component():
    # Pas de key= (ta version Streamlit ne le supporte pas)
    nonce = str(time.time())
    return components.html(
        f"""
        <script>
        (function () {{
          const send = (obj) => {{
            const txt = JSON.stringify(obj);
            window.parent.postMessage(
              {{ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: txt }},
              "*"
            );
          }};

          // nonce: {nonce}

          if (!navigator.geolocation) {{
            send({{ok:false, error:"Geolocation not supported"}});
            return;
          }}

          navigator.geolocation.getCurrentPosition(
            (pos) => send({{ok:true, coords:{{
              latitude: pos.coords.latitude,
              longitude: pos.coords.longitude,
              accuracy: pos.coords.accuracy
            }}}}),
            (err) => send({{ok:false, code: err.code, error: err.message}}),
            {{ enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }}
          );
        }})();
        </script>
        """,
        height=0,
    )

def _ip_city_component():
    nonce = str(time.time())
    return components.html(
        f"""
        <script>
        (function () {{
          const send = (obj) => {{
            const txt = JSON.stringify(obj);
            window.parent.postMessage(
              {{ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: txt }},
              "*"
            );
          }};

          // nonce: {nonce}

          fetch("https://ipapi.co/json/")
            .then(r => r.json())
            .then(d => send({{ok:true, city:d.city, region:d.region, country:d.country_name}}))
            .catch(e => send({{ok:false, error:String(e)}}));
        }})();
        </script>
        """,
        height=0,
    )

def _poll_browser_geo(timeout_s: float = 12.0, max_attempts: int = 25):
    """Lance des reruns jusqu'à récupérer un payload GPS (ou timeout)."""
    if not st.session_state.geo_pending:
        return

    if time.time() - st.session_state.geo_started_at > timeout_s:
        st.session_state.geo_pending = False
        st.session_state.geo_payload = {"ok": False, "error": "timeout_global"}
        return

    if st.session_state.geo_attempt >= max_attempts:
        st.session_state.geo_pending = False
        st.session_state.geo_payload = {"ok": False, "error": "max_attempts"}
        return

    st.session_state.geo_attempt += 1
    raw = _browser_geo_component()

    # raw arrive souvent au rerun suivant → polling
    if raw:
        try:
            st.session_state.geo_payload = json.loads(raw)
        except Exception:
            st.session_state.geo_payload = {"ok": False, "error": "invalid_json", "raw": raw}
        st.session_state.geo_pending = False
        return

    time.sleep(0.25)
    st.rerun()

def _poll_ip_city(timeout_s: float = 8.0, max_attempts: int = 20):
    """Récupère la ville via IP côté navigateur (quasi toujours dispo)."""
    if not st.session_state.ip_pending:
        return

    if time.time() - st.session_state.ip_started_at > timeout_s:
        st.session_state.ip_pending = False
        st.session_state.ip_payload = {"ok": False, "error": "timeout_global"}
        return

    if st.session_state.ip_attempt >= max_attempts:
        st.session_state.ip_pending = False
        st.session_state.ip_payload = {"ok": False, "error": "max_attempts"}
        return

    st.session_state.ip_attempt += 1
    raw = _ip_city_component()

    if raw:
        try:
            st.session_state.ip_payload = json.loads(raw)
        except Exception:
            st.session_state.ip_payload = {"ok": False, "error": "invalid_json", "raw": raw}
        st.session_state.ip_pending = False
        return

    time.sleep(0.25)
    st.rerun()

def detect_city_best_effort() -> tuple[str | None, dict]:
    """
    Option "quasi certaine":
    1) GPS navigateur -> reverse geocode -> ville
    2) sinon ville IP côté navigateur (ipapi) -> ville
    3) sinon None
    """
    dbg = {}

    # 1) Si payload GPS disponible
    geo = st.session_state.geo_payload
    dbg["geo_payload"] = geo

    if geo and geo.get("ok") and "coords" in geo:
        lat = geo["coords"]["latitude"]
        lon = geo["coords"]["longitude"]
        acc = geo["coords"].get("accuracy", 999999)
        dbg["coords"] = {"lat": lat, "lon": lon, "accuracy": acc}

        # Pour la ville, on accepte une précision large
        if acc <= 100000:  # 100 km (PC parfois large)
            city = reverse_city_google(lat, lon) or reverse_city_osm(lat, lon)
            dbg["chosen"] = "browser_gps"
            dbg["city_from_gps"] = city
            if city:
                return city, dbg
        else:
            dbg["gps_ignored"] = f"accuracy too high: {acc}"

    # 2) Sinon, IP côté navigateur (quasi toujours)
    ipd = st.session_state.ip_payload
    dbg["ip_payload"] = ipd
    if ipd and ipd.get("ok"):
        city = ipd.get("city")
        dbg["chosen"] = "browser_ip"
        dbg["city_from_ip"] = city
        if city:
            return city, dbg

    dbg["chosen"] = "none"
    return None, dbg

# =========================
# Business helpers
# =========================
@st.cache_data(ttl=600)
def get_local_weather(city: str) -> str:
    wa = disp.agents["météo"]
    return wa.handle_request(f"météo à {city}")

@st.cache_data(ttl=600)
def get_local_loisirs(city: str) -> str:
    la = disp.agents["loisirs"]
    return la.handle_request(
        f"activités à proximité de {city}, uniquement les titres avec des émojis en lien avec l'activité, ne fais pas de phrases s'il te plaît"
    )

def preprocess_input(prompt: str, cats: list[str], user_city: str | None, geo_allowed: bool) -> str:
    inp = prompt
    if geo_allowed and user_city:
        if "météo" in cats and not re.search(r"\bà\s+\w+", prompt, flags=re.IGNORECASE):
            inp = f"météo à {user_city}"
        if "transport" in cats and not re.search(r"de\s+.+?\s+à\s+.+", prompt, flags=re.IGNORECASE):
            if m := re.search(r"à\s+(.+)", prompt, flags=re.IGNORECASE):
                inp = f"de {user_city} à {m.group(1).strip()}"
    return inp

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Options")

    consent = st.radio(
        "Utiliser votre position (plus précis) ?",
        ["Refuser", "Autoriser"],
        index=0,
        key="geo_consent",
    )
    ctx.geo_permission = (consent == "Autoriser")

    typing = st.toggle("Effet d'écriture", value=True)
    show_local = st.toggle("Afficher infos locales", value=True)

    st.divider()

    manual = st.text_input("Ville (manuel)", value=st.session_state.user_city or "", placeholder="Ex: Valenciennes")
    if st.button("✅ Utiliser la ville manuelle", use_container_width=True):
        if manual.strip():
            set_city(manual.strip())
        else:
            st.warning("Entre une ville.")

    if ctx.geo_permission:
        if st.button("📍 Détecter ma position", use_container_width=True):
            # Reset + lance les deux méthodes (GPS + IP) en parallèle
            st.session_state.geo_pending = True
            st.session_state.geo_started_at = time.time()
            st.session_state.geo_attempt = 0
            st.session_state.geo_payload = None

            st.session_state.ip_pending = True
            st.session_state.ip_started_at = time.time()
            st.session_state.ip_attempt = 0
            st.session_state.ip_payload = None

            st.rerun()

        # Poll GPS d'abord (si possible)
        if st.session_state.geo_pending:
            st.info("Détection GPS en cours… accepte la demande de localisation si elle apparaît.")
            _poll_browser_geo()

        # Poll IP (quasi certain pour une ville)
        if st.session_state.ip_pending:
            st.info("Fallback ville via IP en cours…")
            _poll_ip_city()

        # Quand tout est fini, on calcule la ville
        if (not st.session_state.geo_pending) and (not st.session_state.ip_pending) and (st.session_state.geo_payload or st.session_state.ip_payload):
            city, dbg = detect_city_best_effort()
            st.write(dbg)

            if city:
                set_city(city)
                st.success(f"Ville détectée : {city}")
            else:
                st.error("Impossible de détecter la ville. Utilise la saisie manuelle.")
    else:
        st.caption("Autorise la position pour tenter une géoloc précise. Sinon, utilise la ville manuelle.")

    if st.button("🧹 Réinitialiser", use_container_width=True):
        st.session_state.history = []
        st.session_state.user_city = None
        ctx.location = None
        ctx.city = None

        st.session_state.geo_pending = False
        st.session_state.geo_payload = None
        st.session_state.ip_pending = False
        st.session_state.ip_payload = None
        st.rerun()

# =========================
# Header
# =========================
st.markdown('<div class="chat-title">🧭 Assistant Mobilité Urbaine</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Transport • Météo • Culture • Loisirs — routage automatique (SBERT + fallback)</div>', unsafe_allow_html=True)

user_city = st.session_state.user_city
if user_city:
    st.markdown(f'<div class="small-muted">📍 Ville : <b>{user_city}</b></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="small-muted">📍 Ville : non définie</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# Optional local info
# =========================
if show_local and user_city:
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🌦️ Météo", expanded=False):
            st.write(get_local_weather(user_city))
    with c2:
        with st.expander("🎉 Loisirs", expanded=False):
            st.write(get_local_loisirs(user_city))

# =========================
# Chat (ChatGPT-like)
# =========================
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Écris ta question…")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cats = disp.classify_request(prompt)
    inp = preprocess_input(prompt, cats, user_city, ctx.geo_permission)
    answer = disp.route_request(inp)

    if typing:
        with st.chat_message("assistant"):
            ph = st.empty()
            partial = ""
            for ch in answer:
                partial += ch
                ph.markdown(partial)
                time.sleep(0.004)
        st.session_state.history.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})
