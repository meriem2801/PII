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
from streamlit_js_eval import get_geolocation
import json
import streamlit.components.v1 as components
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

@st.cache_data(ttl=3600)
def ip_geolocate_google() -> str | None:
    """
    IP-based : en local peut être OK, en cloud souvent faux.
    On ne l'utilise qu'en fallback.
    """
    if not gmaps:
        return None
    try:
        geo = gmaps.geolocate()
        lat, lon = geo["location"]["lat"], geo["location"]["lng"]
        return reverse_city_google(lat, lon) or reverse_city_osm(lat, lon)
    except Exception:
        return None

def set_city(city: str | None):
    if city:
        st.session_state.user_city = city
        ctx.location = city
        ctx.city = city
def _geo_component_once():
    """
    Appelle navigator.geolocation.getCurrentPosition() côté navigateur
    et renvoie une string JSON via le channel Streamlit.
    """
    return components.html(
        """
        <script>
        (function () {
          const send = (obj) => {
            const txt = JSON.stringify(obj);
            window.parent.postMessage(
              { isStreamlitMessage: true, type: "streamlit:setComponentValue", value: txt },
              "*"
            );
          };

          if (!navigator.geolocation) {
            send({ok:false, error:"Geolocation not supported"});
            return;
          }

          navigator.geolocation.getCurrentPosition(
            (pos) => send({ok:true, coords:{
              latitude: pos.coords.latitude,
              longitude: pos.coords.longitude,
              accuracy: pos.coords.accuracy
            }}),
            (err) => send({ok:false, code: err.code, error: err.message}),
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
          );
        })();
        </script>
        """,
        height=0,
        key=f"geo_comp_{time.time()}",
    )

def get_geolocation_fallback():
    """
    Retourne un dict du style:
    - {"coords": {"latitude":..., "longitude":..., "accuracy":...}}
    ou None si pas encore dispo / refus / erreur.
    """
    # 1) On tente d'abord streamlit_js_eval (ton actuel)
    try:
        geo = get_geolocation()
        if geo and "coords" in geo:
            return geo
    except Exception:
        pass

    # 2) Fallback: composant HTML/JS (getCurrentPosition)
    raw = _geo_component_once()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    if data.get("ok") and "coords" in data:
        # On renvoie au format attendu par ton code actuel
        return {"coords": data["coords"]}

    return None


def detect_city_hybrid() -> tuple[str | None, dict | None]:
    """
    1) Browser geolocation (si précis)
    2) Sinon fallback IP Google geolocate()
    Retourne (city, debug_dict)
    """
    debug = {}

    # 1) Navigateur
    try:
        geo = get_geolocation_fallback()
        debug["browser_geo"] = geo
        if geo and "coords" in geo:
            lat = geo["coords"]["latitude"]
            lon = geo["coords"]["longitude"]
            acc = geo["coords"].get("accuracy", 999999)
            debug["coords"] = {"lat": lat, "lon": lon, "accuracy": acc}

            # si trop imprécis, on ignore
            if acc <= 50000:
                city = reverse_city_google(lat, lon) or reverse_city_osm(lat, lon)
                if city:
                    debug["chosen"] = "browser"
                    return city, debug
            else:
                debug["browser_ignored"] = f"accuracy too high: {acc}"

    except Exception as e:
        debug["browser_error"] = str(e)

    # 2) Fallback IP Google
    city_ip = ip_geolocate_google()
    debug["chosen"] = "ip_google" if city_ip else "none"
    return city_ip, debug

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
    # debug_geo = st.toggle("Debug géoloc", value=False)

    st.divider()

    manual = st.text_input("Ville (manuel)", value=st.session_state.user_city or "", placeholder="Ex: Valenciennes")
    if st.button("✅ Utiliser la ville manuelle", use_container_width=True):
        if manual.strip():
            set_city(manual.strip())
        else:
            st.warning("Entre une ville.")

    if ctx.geo_permission:
        if st.button("📍 Détecter ma position", use_container_width=True):
            city, dbg = detect_city_hybrid()
            st.write(dbg)
            if city:
                set_city(city)
                st.success(f"Ville détectée : {city}")
            else:
                st.error("Impossible de détecter la ville. Utilise la saisie manuelle.")

            # if debug_geo:
            #     st.write(dbg)
    else:
        st.caption("Autorise la position pour tenter une géoloc précise. Sinon, utilise la ville manuelle.")

    if st.button("🧹 Réinitialiser", use_container_width=True):
        st.session_state.history = []
        st.session_state.user_city = None
        ctx.location = None
        ctx.city = None
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