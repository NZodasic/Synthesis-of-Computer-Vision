import os, time

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def time_str() -> str: return time.strftime('%Y%m%d_%H%M%S')
