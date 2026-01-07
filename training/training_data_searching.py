import os
import json
import re
import random
import logging
from collections import Counter
from dotenv import load_dotenv
import praw
import prawcore
from sklearn.model_selection import train_test_split

# Pour les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _get_streamlit_secrets():
    """
    Retourne st.secrets si streamlit est dispo, sinon None.
    (Import lazy pour rester utilisable en script local sans Streamlit.)
    """
    try:
        import streamlit as st  # type: ignore
        return st.secrets
    except Exception:
        return None


def get_config(key: str, default: str | None = None) -> str | None:
    """
    Récupère une config dans cet ordre:
      1) st.secrets[key]  (Streamlit Cloud - secrets à plat)
      2) st.secrets["reddit"][key] (Streamlit Cloud - secrets dans une section [reddit])
      3) os.environ[key]
      4) default
    """
    secrets = _get_streamlit_secrets()

    # 1) secrets à plat : st.secrets["REDDIT_CLIENT_ID"]
    if secrets is not None:
        try:
            if key in secrets and str(secrets[key]).strip() != "":
                return str(secrets[key]).strip()
        except Exception:
            pass

        # 2) secrets en section : st.secrets["reddit"]["client_id"] etc.
        # On mappe aussi les clés reddit courantes
        section_map = {
            "REDDIT_CLIENT_ID": "client_id",
            "REDDIT_CLIENT_SECRET": "client_secret",
            "REDDIT_USERNAME": "username",
            "REDDIT_PASSWORD": "password",
            "REDDIT_USER_AGENT": "user_agent",
        }
        try:
            if "reddit" in secrets:
                subkey = section_map.get(key, key)
                if subkey in secrets["reddit"] and str(secrets["reddit"][subkey]).strip() != "":
                    return str(secrets["reddit"][subkey]).strip()
        except Exception:
            pass

    # 3) variables d'environnement
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip() != "":
        return env_val.strip()

    return default


class RequestDataset:
    def __init__(self, path, label2id):
        self.samples = []
        self.labels = []
        with open(path, encoding="utf8") as f:
            for line in f:
                item = json.loads(line)
                text = item["text"].strip().replace("\n", " ")
                label = label2id.get(item["label"])
                if label is not None:
                    self.samples.append((text, label))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DataFetcher:
    """Récupère et prépare les données depuis Reddit."""
    _PATTERNS = {
        "transport": re.compile(r"\b(train|bus|voiture|m[ée]tro|covoiturage|taxi)\b", re.I),
        "météo": re.compile(r"\b(m[ée]t[ée]o|pluie|neige|orage|temp[ée]rature)\b", re.I),
        "culture": re.compile(r"\b(livre|film|cin[ée]ma|mus[ée]e)\b", re.I),
        "loisirs": re.compile(
            r"\b(que\s+faire|quoi\s+faire|activit[\wé]*|vacances|voyage|jeux?|randonn[ée]e?|sport|foot|basket|trail|ski|surf|p[êe]che|concert|festival|loisir)\b",
            re.I,
        ),
    }

    def __init__(self, max_per_label=200):
        # Local: charge .env (Streamlit Cloud: ça ne gêne pas)
        load_dotenv()

        # ✅ Récupération compatible local + cloud
        client_id = get_config("REDDIT_CLIENT_ID")
        client_secret = get_config("REDDIT_CLIENT_SECRET")
        username = get_config("REDDIT_USERNAME")
        password = get_config("REDDIT_PASSWORD")
        user_agent = get_config("REDDIT_USER_AGENT", "scraper-bot")

        missing = [k for k, v in {
            "REDDIT_CLIENT_ID": client_id,
            "REDDIT_CLIENT_SECRET": client_secret,
            "REDDIT_USERNAME": username,
            "REDDIT_PASSWORD": password,
        }.items() if not v]

        if missing:
            raise RuntimeError(
                "Variables Reddit manquantes: " + ", ".join(missing) + "\n\n"
                "✅ En local: mets-les dans un fichier .env\n"
                "✅ Sur Streamlit Cloud: Settings -> Secrets, exemple:\n"
                "REDDIT_CLIENT_ID = \"...\"\n"
                "REDDIT_CLIENT_SECRET = \"...\"\n"
                "REDDIT_USERNAME = \"...\"\n"
                "REDDIT_PASSWORD = \"...\"\n"
                "REDDIT_USER_AGENT = \"scraper-bot\"\n\n"
                "Ou en section:\n"
                "[reddit]\n"
                "client_id = \"...\"\n"
                "client_secret = \"...\"\n"
                "username = \"...\"\n"
                "password = \"...\"\n"
                "user_agent = \"scraper-bot\""
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent,
        )

        self.max_per_label = max_per_label
        self.seen = set()

    def clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def fetch_label(self, subreddits, keywords, label) -> list[dict]:
        collected = []
        logging.info(f"On récupère données pour '{label}' dans {subreddits} avec mot clef {keywords}")
        query = " OR ".join(keywords)

        # passe 1 : sort="new"
        for sub in subreddits:
            try:
                posts = self.reddit.subreddit(sub).search(query, sort="new", limit=1500)
            except prawcore.exceptions.Forbidden:
                logging.warning(f"→ Subreddit '{sub}' inaccessible (403), on zappe.")
                continue
            except Exception as e:
                logging.warning(f"→ Erreur sur '{sub}': {e}, on zappe.")
                continue

            try:
                for post in posts:
                    title = self.clean(post.title)
                    if title in self.seen or "?" not in title or len(title.split()) < 4:
                        continue
                    if not self._PATTERNS[label].search(title):
                        continue
                    self.seen.add(title)
                    collected.append({"text": title, "label": label})
                    if len(collected) >= self.max_per_label:
                        break
            except prawcore.exceptions.Forbidden:
                logging.warning(f"→ 403 pendant récup des datas de '{sub}', on zappe.")
                continue
            except Exception as e:
                logging.warning(f"→ Erreur pendant l'itération des posts de '{sub}': {e}, on zappe.")
                continue

            if len(collected) >= self.max_per_label:
                break

        # passe 2 pour 'loisirs' si insuffisant
        if label == "loisirs" and len(collected) < self.max_per_label:
            logging.info("Pas assez de data 'loisirs' → on y retourne.")
            for sub in subreddits:
                try:
                    posts = self.reddit.subreddit(sub).search(query, sort="top", time_filter="year", limit=1500)
                except prawcore.exceptions.Forbidden:
                    logging.warning(f"→ Subreddit '{sub}' inaccessible (403), on zappe.")
                    continue
                except Exception as e:
                    logging.warning(f"→ Erreur sur '{sub}': {e}, skipping.")
                    continue

                try:
                    for post in posts:
                        title = self.clean(post.title)
                        if title in self.seen or len(title.split()) < 4:
                            continue
                        if not self._PATTERNS[label].search(title):
                            continue
                        self.seen.add(title)
                        collected.append({"text": title, "label": label})
                        if len(collected) >= self.max_per_label:
                            break
                except prawcore.exceptions.Forbidden:
                    logging.warning(f"→ 403 pendant récup de '{sub}', on zappe.")
                    continue
                except Exception as e:
                    logging.warning(f"→ Erreur pendant l'itération des posts de '{sub}': {e}, on zappe.")
                    continue

                if len(collected) >= self.max_per_label:
                    break

        logging.info(f"Recuperation de {len(collected)} exemples pour '{label}'")
        return collected

    def run(self, themes: dict, extra_paths: list[str] | None = None) -> tuple[list, list]:
        data = []
        for label, cfg in themes.items():
            data.extend(self.fetch_label(cfg["subreddits"], cfg["keywords"], label))

        # chargement des fichiers d'extra données
        if extra_paths:
            for extra in extra_paths:
                if os.path.exists(extra):
                    logging.info(f"Je récupère les questions supplémentaires {extra}")
                    with open(extra, encoding="utf-8") as f:
                        for line in f:
                            item = json.loads(line)
                            text = self.clean(item["text"])
                            if text not in self.seen:
                                data.append({"text": text, "label": item["label"]})
                                self.seen.add(text)

        counts = Counter(d["label"] for d in data)
        for lbl, cnt in counts.items():
            if cnt < 2:
                logging.warning(f"Seuls {cnt} exemples pour '{lbl}' – duplication.")
                sample = next(d for d in data if d["label"] == lbl)
                for _ in range(2 - cnt):
                    data.append(sample.copy())

        random.shuffle(data)

        counts = Counter(d["label"] for d in data)
        if min(counts.values()) < 2:
            logging.warning("Split train/val (80/20) sans stratification (classes trop petites).")
            train, val = train_test_split(data, test_size=0.2, random_state=42)
            return train, val

        # Si on a assez d'exemples, on peut stratifier
        train, val = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=[d["label"] for d in data],
        )
        return train, val

    def save(self, data: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        logging.info(f"On a sauvé {len(data)} enregistrements à {path}")


if __name__ == "__main__":
    THEMES = {
        "transport": {"subreddits": ["france", "voyage"], "keywords": ["train", "bus", "voiture", "avion"]},
        "météo": {"subreddits": ["meteo", "montagne"], "keywords": ["météo", "orage", "neige", "pluie", "température"]},
        "culture": {
            "subreddits": ["Livres", "cinema", "culture_generale", "france"],
            "keywords": ["livre", "film", "musée", "politique", "guerre", "conflit", "histoire"],
        },
        "loisirs": {
            "subreddits": ["jeuxvideo", "Voyage", "Concerts", "Festival", "hiking", "running", "ski", "sports", "france"],
            "keywords": [
                "jeux", "vacances", "activité", "activites", "randonnée", "sport", "foot", "basket", "trail",
                "pêche", "surf", "concert", "festival", "loisir"
            ],
        },
    }

    fetcher = DataFetcher(max_per_label=1000)
    train, val = fetcher.run(
        THEMES,
        extra_paths=[
            "questions_meteo.jsonl",
            "questions_loisir.jsonl",
            "questions_transport.jsonl",
        ],
    )
    fetcher.save(train, "train.jsonl")
    fetcher.save(val, "val.jsonl")
