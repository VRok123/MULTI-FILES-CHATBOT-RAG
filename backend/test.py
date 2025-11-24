from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

print("Service JSON Path:", os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
