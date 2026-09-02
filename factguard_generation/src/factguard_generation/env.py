import dotenv
import os
import pathlib

dotenv.load_dotenv()

VLLM_API_URL = os.environ.get(
    "VLLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions"
)
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", None)
EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL", "http://127.0.0.1:8081"
)
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

FACTGUARD_DATA_DIR = pathlib.Path(
    os.environ.get(
        "FACTGUARD_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "data",
    )
).expanduser()
