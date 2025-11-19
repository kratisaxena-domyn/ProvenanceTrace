import os
import json

def get_hf_token():
    config_path = "../config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("HF_TOKEN", None)
        except Exception:
            return None
    return os.environ.get("HF_TOKEN", None)