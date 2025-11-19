import json
import os
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config

def get_api_key(service="openai"):
    """Get API key for a service"""
    config = load_config()
    return config["api_keys"].get(service, "Model_API_PLACEHOLDER")

def get_llm_client_config():
    """Get LLM client configuration"""
    config = load_config()
    return config.get("llm_client", {
        "base_url": "<Your_Model_API_Endpoint_URL>",
        "api_key": "EMPTY",
        "model_name": "iGenius-AI-Team/Domyn-Small"
    })

def get_model_settings():
    """Get model configuration"""
    config = load_config()
    return config.get("model_settings", {})