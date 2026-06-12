import os
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = {}

def _get(key: str, default: str = "") -> str:
    return os.getenv(key) or _secrets.get(key, default)
