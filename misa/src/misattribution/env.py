import dotenv
import os
import pathlib

dotenv.load_dotenv()

VLLM_API_URL = os.environ["VLLM_API_URL"]
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", None)

MISA_DATA_DIR = os.environ.get("MISA_DATA_DIR", None)
if MISA_DATA_DIR:
    MISA_DATA_DIR = pathlib.Path(MISA_DATA_DIR)
