import streamlit as st
from dotenv import load_dotenv
import os
import googlemaps
import re
import types
import requests
from agents.dispatcher import Dispatcher
import time
import html

# On configure la page
st.set_page_config(page_title="Assistant Mobilité Urbaine", layout="wide")

# Env vars
load_dotenv()

# Histoire du chat
if "history" not in st.session_state:
    st.session_state.history = []

# Dispatcher + context
@st.cache_resource
def get_dispatcher():
    disp = Dispatcher()
    disp.context = types.SimpleNamespace(
        location=None, geo_permission=False, city=None
    )
    return disp

disp = get_dispatcher()
ctx  = disp.context

# Google Maps client -- cela va nous permettre d'utiliser la géolocalisation (si on a l'accord de l'utilisateur)
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

# BARRE A GAUCHE
with st.sidebar:
    st.header("Options")
    consent = st.radio(
        "Autorisez-vous l'accès à votre position pour le lieu par défaut ?",
        ["Refuser", "Autoriser"],
        index=0, key="geo_consent"
    )
    ctx.geo_permission = (consent == "Autoriser")
    st.write(f"Géolocalisation : {'✅ Autorisée' if ctx.geo_permission else '❌ Refusée'}")

    if st.button("Réinitialiser la conversation"):
        st.session_state.clear()
        get_dispatcher.clear()
        st.experimental_rerun()

# GEOLOCALISATION
def do_geolocate():
    try:
        geo = gmaps.geolocate()
        lat, lon = geo["location"]["lat"], geo["location"]["lng"]
        rev = gmaps.reverse_geocode((lat, lon))
        for comp in rev[0].get("address_components", []):
            if "locality" in comp["types"]:
                return comp["long_name"]
        return rev[0].get("formatted_address")
    except:
        try:
            geo = gmaps.geolocate()
            lat, lon = geo["location"]["lat"], geo["location"]["lng"]
            osm = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format":"json","lat":lat,"lon":lon,"zoom":10},
                headers={"User-Agent":"mobility-app"}
            ).json().get("address", {})
            return osm.get("city") or osm.get("town") or osm.get("village")
        except:
            return None

if ctx.geo_permission and ctx.location is None:
    city = do_geolocate()
    ctx.city     = city
    ctx.location = city or None

# Page principale
col_info, col_chat = st.columns([1, 2])

with col_info:
    st.subheader("Infos locales")
    if ctx.geo_permission and ctx.location:
        st.markdown(f"**Localisation :** {ctx.location}")
        # partie météo -- chargé automatiquement par rapport à la localisation
        wa    = disp.agents["météo"]
        meteo = wa.handle_request(f"météo à {ctx.location}")
        icon  = "❓"
        if "clair"   in meteo: icon = "☀️"
        elif "nuage" in meteo: icon = "☁️"
        elif "pluie" in meteo: icon = "🌧️"
        elif "neige" in meteo: icon = "❄️"
        elif "orage" in meteo: icon = "⛈️"
        st.markdown(f"### Météo actuelle {icon}")
        st.write(meteo)

        # partie loisir -- chargé automatiquement par rapport à la localisation
        la = disp.agents["loisirs"]
        loisirs_txt = la.handle_request(
            f"activités à proximité de {ctx.location}, uniquement les titres avec des émojis en lien avec l'activité s'il te plaît"

        )
        st.markdown("### Loisirs à proximité 🎉")
        items = re.findall(r"(?m)^\s*(\d+\.\s*[^\n]+)", loisirs_txt)
        if items:
            for itm in items[:3]:
                st.write(f" {itm.strip()}")
        else:
            first = loisirs_txt.splitlines()[0].strip()
            st.write(f" {first}")
    else:
        st.write("Géolocalisation refusée ou introuvable.")

with col_chat:
    st.subheader("Conversation")

    #style
    st.markdown("""
        <style>
        /* boite de conversation */
        #chat-box{
            height: 70vh;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 1rem;
            background: #fafafa;
        }
        .msg{ margin-bottom:.75rem; }
        .user strong{ color:#1f77b4; }
        .bot  strong{ color:#333;    }
        </style>
        """, unsafe_allow_html=True)

    box = st.empty()

    def render():
        """Construit la chatBox et la met dans le placeholder"""
        lines = ['<div id="chat-box">']
        for role, txt in st.session_state.history:
            css = "user" if role == "Vous" else "bot"
            lines.append(
                f'<div class="msg {css}"><strong>{role} :</strong> '
                f'{html.escape(txt)}</div>'
            )
        lines.append('</div>')

        box.markdown("\n".join(lines), unsafe_allow_html=True)

    render()

    # barre de questions
    prompt = st.chat_input("Posez votre question…")

    # traitement du prompt
    if prompt:
        st.session_state.history.append(("Vous", prompt))
        render()                     # on va montrer la question tout de suite

        # pre-traitement puis appel des agents
        inp  = prompt
        cats = disp.classify_request(prompt)
        if "météo" in cats and ctx.geo_permission and ctx.location \
                and not re.search(r"\bà\s+\w+", prompt):
            inp = f"météo à {ctx.location}"
        if "transport" in cats and ctx.geo_permission \
                and not re.search(r"de\s+.+?\s+à\s+.+", prompt):
            if m := re.search(r"à\s+(.+)", prompt):
                inp = f"de {ctx.location} à {m.group(1).strip()}"

        answer = disp.route_request(inp)

        # Je voulais que la réponse fasse un effet type Chat GPT ou autre IA conversationnelle où le texte apparaît petit à petit
        partial = ""
        for ch in answer:
            partial += ch
            if st.session_state.history and st.session_state.history[-1][0] == "Assistant":
                st.session_state.history[-1] = ("Assistant", partial)
            else:
                st.session_state.history.append(("Assistant", partial))
            render()
            time.sleep(0.01) #le sleep permet de rendre cet effet

        render()