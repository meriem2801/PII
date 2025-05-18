import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SNCF_API_KEY= os.getenv("SNCF_API_KEY")
