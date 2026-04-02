import json
import os
from typing import Optional, Dict, Any, List

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")

def save_profile(field: str, data: Dict[str, Any]):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    file_path = os.path.join(PROFILES_DIR, f"{field}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_profile(field: str) -> Optional[Dict[str, Any]]:
    file_path = os.path.join(PROFILES_DIR, f"{field}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    
    # Fallback to general.json
    general_path = os.path.join(PROFILES_DIR, "general.json")
    if os.path.exists(general_path):
        with open(general_path, "r") as f:
            return json.load(f)
            
    return None

def list_profiles() -> List[str]:
    if not os.path.exists(PROFILES_DIR):
        return []
    
    profiles = []
    for filename in os.listdir(PROFILES_DIR):
        if filename.endswith(".json"):
            profiles.append(filename.replace(".json", ""))
    return profiles
