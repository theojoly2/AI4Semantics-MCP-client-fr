import os
from asyncio import run as asyncio_run
import streamlit as st
from config import load_config
from dotenv import load_dotenv

load_dotenv()

config = load_config()
SERVER = config["MCP-server"]["local"]

from chat_interface import data_modelling_chat_tab

try:
    st.session_state.setdefault("add_env", True)
    if st.session_state["add_env"]:
        st.session_state["add_env"] = False
        os.environ.update(st.secrets)
except Exception as e:
    st.warning(f"Could not apply secrets to environment: {e}")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Réduit fortement l'espace vide en haut */
div[data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
    padding-bottom: 0rem !important;
}

/* Fallback pour certaines versions */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0rem !important;
}

/* Rend le header transparent et plus discret */
header.stAppHeader {
    background-color: transparent;
}

/* Supprime la ligne décorative rouge du haut si elle gêne */
div[data-testid="stDecoration"] {
    display: none;
}

/* Optionnel : réduit aussi le haut de la sidebar */
[data-testid="stSidebarHeader"] {
    height: 2rem;
}
</style>
""", unsafe_allow_html=True)

tab1, *_ = st.tabs(["Data Model chat"])

with tab1:
    try:
        data_modelling_chat_tab(server=SERVER)
    except Exception as e:
        try:
            with st.status("The UI encountered an unexpected error.", expanded=True, state="error") as status:
                st.write(str(e))
                st.write(
                    "**Ce que vous pouvez faire maintenant :**\n"
                    "1) Vérifiez vos saisies et corrigez le bug si possible.\n"
                    "2) Relancez l’interface utilisateur.\n"
                    "3) Si l’erreur persiste, contactez l’équipe technique à l’adresse **theo.joly2@developpement-durable.gouv.fr**."
                )
                status.update(label="Action required", state="error")
        except Exception:
            st.error("The UI encountered an unexpected error.")
            st.write(str(e))
            st.write(
                "**Ce que vous pouvez faire maintenant :**\n"
                "1) Vérifiez vos saisies et corrigez le bug si possible.\n"
                "2) Relancez l’interface utilisateur.\n"
                "3) Si l’erreur persiste, contactez l’équipe technique à l’adresse **theo.joly2@developpement-durable.gouv.fr**."
            )
