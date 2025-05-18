import re
import requests

class WeatherAgent:
    def __init__(self):
        # Endpoints pour la géocodification et la météo via Open-Meteo
        self.geocoding_api_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_api_url = "https://api.open-meteo.com/v1/forecast"

    def extract_city(self, user_input):
        """
        Extrait le nom d'une ville depuis la requête en recherchant
        les prépositions 'à' ou 'pour' suivies d'un nom de ville.
        """
        match = re.search(r"(?:à|pour)\s+([A-Za-zÀ-ÖØ-öø-ÿ\s\-]+)", user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def get_coordinates(self, city):
        """
        Utilise le service de géocodage d'Open-Meteo pour obtenir
        les coordonnées (latitude, longitude) de la ville.
        """
        params = {
            "name": city,
            "count": 1,
            "language": "fr",
            "format": "json"
        }
        response = requests.get(self.geocoding_api_url, params=params)
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return result["latitude"], result["longitude"]
        else:
            return None, None

    def map_weather_code(self, code):
        """
        Mappe les codes météo d'Open-Meteo à une description textuelle simplifiée.

        """
        mapping = {
            0: "clair",
            1: "principalement clair",
            2: "partiellement nuageux",
            3: "couvert",
            45: "brouillard",
            48: "brouillard givrant",
            51: "bruine légère",
            53: "bruine modérée",
            55: "bruine dense",
            56: "bruine verglaçante légère",
            57: "bruine verglaçante dense",
            61: "pluie légère",
            63: "pluie modérée",
            65: "pluie forte",
            66: "pluie verglaçante légère",
            67: "pluie verglaçante forte",
            71: "neige faible",
            73: "neige modérée",
            75: "neige forte",
            77: "neige en grains",
            80: "averses de pluie légère",
            81: "averses de pluie modérée",
            82: "averses de pluie forte",
            85: "averses de neige légère",
            86: "averses de neige forte",
            95: "orage",
            96: "orage avec grêle légère",
            99: "orage avec grêle forte"
        }
        return mapping.get(code, "indéterminé")

    def handle_request(self, user_input):
        city = self.extract_city(user_input)
        if not city:
            return "Veuillez préciser la ville pour laquelle vous souhaitez connaître la météo."

        lat, lon = self.get_coordinates(city)
        if lat is None or lon is None:
            return f"Impossible de trouver les coordonnées pour la ville {city}."

        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "Europe/Paris"
        }
        try:
            response = requests.get(self.weather_api_url, params=params)
            data = response.json()
            if "current_weather" not in data:
                return "Erreur lors de la récupération des données météo."
            current_weather = data["current_weather"]
            temperature = current_weather["temperature"]
            windspeed = current_weather["windspeed"]
            weathercode = current_weather["weathercode"]
            weather_description = self.map_weather_code(weathercode)
            result = (f"À {city.capitalize()}, le temps est {weather_description}, "
                      f"la température est de {temperature}°C et la vitesse du vent est de {windspeed} km/h.")
            return result
        except Exception as e:
            return "Erreur lors de la récupération des données météo."
