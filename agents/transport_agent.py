import os
import re
from datetime import datetime

from openai import OpenAI
import googlemaps

class TransportAgent:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.gmaps = googlemaps.Client(
            key=os.getenv("GOOGLE_MAPS_API_KEY"),
            timeout=10
        )

    def extract_parameters(self, text: str):
        """
        Extrait 'origin' et 'destination' du texte
        en cherchant la forme "de X à Y".
        Si échec, tente de trouver deux noms propres (commençant par majuscule,
        sauf le premier mot) comme lieux.
        """
        # Tentative classique "de X à Y"
        match = re.search(r'de\s+([^\n]+?)\s+à\s+([^\n]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Fallback : noms propres en capitales (sauf premier mot)
        # On trouve tous les mots commençant par une majuscule
        tokens = re.findall(r"\b([A-Z][a-zA-ZÀ-ÖØ-öø-ÿ'\-]+)\b", text)
        # Ignorer le premier mot si c'est en majuscule (ex: "Je", "Vous")
        if len(tokens) > 1:
            # On prend les deux premiers mots restants
            # ce n'est pas optimal mais j'avoue ne pas avoir trouvé mieux sur le coup
            origin, destination = tokens[:2]
            return origin, destination

        return None, None

    def classify_request(self, user_input: str) -> str:
        """
        Retourne 'ITINERARY' si la requête est un itinéraire, sinon 'GENERAL'.
        """
        prompt = (
            "Vous êtes un classificateur qui décide si la requête de l'utilisateur "
            "est une demande d'itinéraire (« de A à B ») ou une question générale sur les transports.\n"
            "Répondez strictement par ITINERARY ou GENERAL."
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",  "content": prompt},
                {"role": "user",    "content": user_input}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip().upper()

    def reformulate(self, user_input: str) -> str:
        """
        Transforme la phrase en "de X à Y" si possible, sinon renvoie ''.
        """
        prompt = (
            "Transformez la phrase de l'utilisateur en une forme exacte « de X à Y ». "
            "Si non pertinent, renvoyez une chaîne vide."
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",  "content": prompt},
                {"role": "user",    "content": user_input}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    def handle_request(self, user_input: str) -> str:
        # Classification
        kind = self.classify_request(user_input)
        if kind != "ITINERARY":
            # question générale, on délègue à OpenAI --- type "Quel est le moyen de transport le + écologique"
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",  "content": "Vous êtes un expert en transport. Répondez clairement à la question."},
                    {"role": "user",    "content": user_input}
                ]
            )
            return resp.choices[0].message.content.strip()

        # Extraction ou reformulation
        origin, destination = self.extract_parameters(user_input)
        if not (origin and destination):
            reformu = self.reformulate(user_input)
            origin, destination = self.extract_parameters(reformu)

        if not (origin and destination):
            return (
                "Désolé, je n'ai pas compris d'où à où. "
                "Merci d'indiquer votre itinéraire sous la forme « de X à Y ». "
            )

        # Appel Google Maps en français
        now = datetime.now()
        try:
            routes = self.gmaps.directions(
                origin,
                destination,
                mode="transit",
                departure_time=now,
                alternatives=True,
                language="fr"
            )
        except Exception as e:
            return f"Erreur API Google Maps : {e}"

        if not routes:
            return f"Aucun itinéraire trouvé entre « {origin} » et « {destination} »."

        # Mise en forme
        lines = [
            f"Itinéraires de {origin} → {destination}",
            f"*Départ prévu à {now.strftime('%H:%M')}*"
        ]
        for idx, route in enumerate(routes, start=1):
            leg = route["legs"][0]
            dur_tot = leg["duration"]["text"]
            lines.append(f"\n### Itinéraire #{idx} — durée totale : {dur_tot}")
            for step in leg["steps"]:
                mode = step["travel_mode"]
                if mode == "WALKING":
                    dist = step["distance"]["text"]
                    d = step["duration"]["text"]
                    lines.append(f"- 🚶 **À pied** : {dist} ({d})")
                elif mode == "TRANSIT":
                    td = step["transit_details"]
                    li = td.get("line", {})
                    name = li.get("short_name") or li.get("name") or "Ligne"
                    dep = td["departure_stop"]["name"]
                    arr = td["arrival_stop"]["name"]
                    dep_t = td["departure_time"]["text"]
                    arr_t = td["arrival_time"]["text"]
                    nst = td.get("num_stops", "?")
                    lines.append(
                        f"- 🚆 **{name}** ({nst} arrêts) : {dep} → {arr} "
                        f"({dep_t}–{arr_t})"
                    )
                else:
                    instr = re.sub(r"<[^>]+>", "", step.get("html_instructions", ""))
                    d = step["duration"]["text"]
                    lines.append(f"- ❓ **{mode}** : {instr} ({d})")
        return "\n".join(lines)
