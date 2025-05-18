from __future__ import annotations
import functools
import torch
import logging
import re
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from agents.transport_agent import TransportAgent
from agents.weather_agent   import WeatherAgent
from agents.culture_agent   import CultureAgent
from agents.loisirs_agent   import LoisirsAgent

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")

class Dispatcher:
    """Route les requêtes vers quatre agents (transport, météo, culture, loisirs)."""

    # Regex pour fallback si SBERT renvoie None : c'est des mots clefs si jamais on a aucune classe
    _KEYWORDS = {
        "transport": r"\b(bus|métro|train|tram|rer|itinéraire|trajet|covoiturage|taxi|aller|voyager|prendre)\b",
        "météo":     r"\b(météo|pluie|neige|soleil|orage|vent|température|prévisions|temps)\b",
        "culture":   r"\b(historiq|patrimoine|monument|musée|château|architecte|histoire|guerre)\b",
        "loisirs":   r"\b(concert|exposition|festival|sortie|loisir|événement|spectacle)\b",
    }

    def __init__(
            self,
            model_path: str = "checkpoints/dispatcher_sbert.pt",
            threshold: float = 0.50,
            secondary_threshold: float = 0.35
    ):
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold

        # Chargement du checkpoint fine-tune par le finetune_dispacther.py
        ckpt = torch.load(model_path, map_location="cpu")
        self.label2id = ckpt["label2id"]
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        # Backbone SBERT
        self.backbone = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.backbone.load_state_dict(ckpt["sbert"])
        self.backbone.eval()

        # Tête de classification
        dim = self.backbone.get_sentence_embedding_dimension()
        self.clf = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, len(self.label2id)),
        )
        self.clf.load_state_dict(ckpt["clf"])
        self.clf.eval()

        # Agents métiers
        self.agents = {
            "transport": TransportAgent(),
            "météo":     WeatherAgent(),
            "culture":   CultureAgent(),
            "loisirs":   LoisirsAgent(),
        }

        # Pré-compile les regex fallback
        self._kw_regex = {
            lbl: re.compile(pat, re.IGNORECASE)
            for lbl, pat in self._KEYWORDS.items()
        }

    @functools.lru_cache(maxsize=256)
    def _encode(self, text: str):
        with torch.no_grad():
            return self.backbone.encode(text, convert_to_tensor=True)

    def _sbert_predict(self, text: str) -> Tuple[Optional[str], float, List[str]]:
        emb = self._encode(text)
        with torch.no_grad():
            logits = self.clf(emb)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)

        idx_main = int(torch.argmax(probs).item())
        score    = float(probs[idx_main])
        label    = self.id2label[idx_main]

        secondaries = [
            self.id2label[i]
            for i, p in enumerate(probs)
            if i != idx_main and p >= self.secondary_threshold
        ]

        logging.debug(
            f"[SBERT] '{text}' → main: {label} ({score:.2f}), secondaries: {secondaries}"
        )
        return label, score, secondaries

    def _keyword_fallback(self, text: str) -> Optional[str]:
        for lbl, regex in self._kw_regex.items():
            if regex.search(text):
                logging.debug(f"[Fallback kw] '{text}' → {lbl}")
                return lbl
        return None

    def classify_request(self, text: str) -> List[str]:
        main, score, secondaries = self._sbert_predict(text)

        # Si score SBERT trop bas (< threshold), tenter fallback par mot clef
        if score < self.threshold:
            kw = self._keyword_fallback(text)
            if kw:
                logging.debug(f"[Score<seuil] '{text}' fallback → {kw}")
                return [kw]
            logging.debug(f"[Score<seuil mais prise SBERT] '{text}' → {main}")

        return [main] + secondaries

    def route_request(self, user_input: str) -> str:
        logging.info(f"[User] {user_input}")
        cats = self.classify_request(user_input)
        logging.info(f"[Cats] {cats}")

        output = []
        for cat in cats:
            agent = self.agents.get(cat)
            if not agent:
                logging.error(f"Aucun agent pour '{cat}'")
                continue
            try:
                logging.debug(f"→ appel agent '{cat}'")
                resp = agent.handle_request(user_input)
            except Exception as e:
                logging.exception(f"Erreur agent '{cat}'")
                resp = f"[Erreur] échec de traitement : {e}"
            output.append(f"[{cat.capitalize()}] {resp}")

        return "\n".join(output)
