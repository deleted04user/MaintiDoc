import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets

import streamlit as st


def identify():
    accounts_file = os.getenv("MAINTIDOC_ACCOUNTS")
    if not accounts_file:
        return "local", "technician"
    with open(accounts_file, encoding="utf-8-sig") as handle:
        accounts = json.load(handle)
    user = st.session_state.get("account")
    if user not in accounts:
        with st.form("login"):
            name = st.text_input("Compte")
            key = st.text_input("Clé personnelle", type="password")
            if st.form_submit_button("Connexion"):
                record = accounts.get(name, {})
                digest = hashlib.sha256(key.encode()).hexdigest()
                if hmac.compare_digest(record.get("sha256", ""), digest):
                    st.session_state.account = name
                    st.rerun()
                st.error("Identifiants invalides.")
        st.stop()
    role = accounts[user]["role"]
    if role not in ("reader", "technician"):
        st.error("Accès refusé.")
        st.stop()
    if st.sidebar.button("Déconnexion"):
        st.session_state.clear()
        st.rerun()
    return user, role


if __name__ == "__main__":
    folder = Path(".maintidoc")
    folder.mkdir(exist_ok=True)
    keys = {r: secrets.token_urlsafe(32) for r in ("reader", "technician")}
    with (folder / "accounts.json").open("x", encoding="utf-8") as handle:
        json.dump({r: {"role": r, "sha256": hashlib.sha256(k.encode()).hexdigest()}
                   for r,k in keys.items()}, handle, indent=2)
    with (folder / "login-keys.json").open("x", encoding="utf-8") as handle:
        json.dump(keys, handle, indent=2)
