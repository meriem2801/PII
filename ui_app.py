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
import requests
import streamlit as st
import googlemaps
from dotenv import load_dotenv

from streamlit_js_eval import get_geolocation, streamlit_js_eval
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

# --- états pour polling ---
if "geo_pending" not in st.session_state:
    st.session_state.geo_pending = False
if "geo_deadline" not in st.session_state:
    st.session_state.geo_deadline = 0.0
if "geo_attempts" not in st.session_state:
    st.session_state.geo_attempts = 0
if "geo_data" not in st.session_state:
    st.session_state.geo_data = None

if "ip_pending" not in st.session_state:
    st.session_state.ip_pending = False
if "ip_deadline" not in st.session_state:
    st.session_state.ip_deadline = 0.0
if "ip_attempts" not in st.session_state:
    st.session_state.ip_attempts = 0
if "ip_data" not in st.session_state:
    st.session_state.ip_data = None

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
# ✅ Ville via IP côté navigateur (quasi certain)
# =========================
def get_city_from_ip_browser():
    """
    Exécute un fetch côté navigateur via streamlit_js_eval.
    Retour attendu: dict {ok, city, region, country} ou None (si pas encore dispo).
    """
    js = """
    (async () => {
      try {
        const r = await fetch("https://ipapi.co/json/");
        const d = await r.json();
        return { ok: true, city: d.city, region: d.region, country: d.country_name };
      } catch(e) {
        return { ok: false, error: String(e) };
      }
    })()
    """
    try:
        return streamlit_js_eval(js_expressions=js, key=f"ip_eval_{st.session_state.ip_attempts}", want_output=True)
    except Exception:
        return None

# =========================
# Détection ville "best effort" (GPS -> IP browser -> None)
# =========================
def detect_city_best_effort() -> tuple[str | None, dict]:
    dbg = {"chosen": "none"}

    # 1) GPS navigateur
    geo = st.session_state.geo_data
    dbg["browser_geo"] = geo
    if geo and "coords" in geo:
        lat = geo["coords"]["latitude"]
        lon = geo["coords"]["longitude"]
        acc = geo["coords"].get("accuracy", 999999)
        dbg["coords"] = {"lat": lat, "lon": lon, "accuracy": acc}

        # Pour juste la ville, on accepte large
        if acc <= 100000:
            city = reverse_city_google(lat, lon) or reverse_city_osm(lat, lon)
            if city:
                dbg["chosen"] = "browser_gps"
                return city, dbg
        else:
            dbg["gps_ignored"] = f"accuracy too high: {acc}"

    # 2) IP côté navigateur
    ipd = st.session_state.ip_data
    dbg["ip_browser"] = ipd
    if ipd and isinstance(ipd, dict) and ipd.get("ok"):
        city = ipd.get("city")
        if city:
            dbg["chosen"] = "browser_ip"
            return city, dbg

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
            # reset + lance polling GPS + polling IP
            st.session_state.geo_pending = True
            st.session_state.geo_deadline = time.time() + 12
            st.session_state.geo_attempts = 0
            st.session_state.geo_data = None

            st.session_state.ip_pending = True
            st.session_state.ip_deadline = time.time() + 8
            st.session_state.ip_attempts = 0
            st.session_state.ip_data = None

            st.rerun()

        # ---- Poll GPS ----
        if st.session_state.geo_pending:
            st.info("Détection GPS en cours… accepte la demande de localisation si elle apparaît.")
            if time.time() > st.session_state.geo_deadline or st.session_state.geo_attempts > 25:
                st.session_state.geo_pending = False
            else:
                st.session_state.geo_attempts += 1
                geo = get_geolocation()
                if geo and "coords" in geo:
                    st.session_state.geo_data = geo
                    st.session_state.geo_pending = False
                else:
                    time.sleep(0.25)
                    st.rerun()

        # ---- Poll IP (browser) ----
        if st.session_state.ip_pending:
            st.info("Fallback ville via IP en cours…")
            if time.time() > st.session_state.ip_deadline or st.session_state.ip_attempts > 20:
                st.session_state.ip_pending = False
            else:
                st.session_state.ip_attempts += 1
                ipd = get_city_from_ip_browser()
                if ipd and isinstance(ipd, dict):
                    st.session_state.ip_data = ipd
                    st.session_state.ip_pending = False
                else:
                    time.sleep(0.25)
                    st.rerun()

        # ---- Résultat ----
        if (not st.session_state.geo_pending) and (not st.session_state.ip_pending) and (st.session_state.geo_data or st.session_state.ip_data):
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
        st.session_state.geo_data = None
        st.session_state.ip_pending = False
        st.session_state.ip_data = None
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
