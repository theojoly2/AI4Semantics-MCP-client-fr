import os
from asyncio import run as asyncio_run
import streamlit as st
from config import load_config
from dotenv import load_dotenv

# CRITICAL: charge le .env avant tout import de OpenAIClient
load_dotenv()

config = load_config()

SERVER = config["MCP-server"]["local"]

from chat_interface import data_modelling_chat_tab  # keep your original import style

# One-time secrets → env hydration (guarded)
try:
    st.session_state.setdefault('add_env', True)
    if st.session_state['add_env']:
        st.session_state['add_env'] = False
        os.environ.update(st.secrets)  # if st.secrets present, update env
except Exception as e:
    # Non-fatal; app can continue
    st.warning(f"Could not apply secrets to environment: {e}")

# Set layout
st.set_page_config(layout="wide")

# Tabs
tab1, *_ = st.tabs(["Data Model chat"])

with tab1:
    try:
        asyncio_run(data_modelling_chat_tab(server=SERVER))
    except Exception as e:
        # Ensure user-friendly error at top-level
        try:
            # Preferred UX if supported
            with st.status("The UI encountered an unexpected error.", expanded=True, state="error") as status:
                st.write(str(e))
                st.write(
                    "**What you can do now:**\n"
                    "1) Review your inputs and correct the bug if possible.\n"
                    "2) Re-launch the UI.\n"
                    "3) If the error keeps happening, contact the tech team at **emilien.caudron@pwc.com**."
                )
                status.update(label="Action required", state="error")
        except Exception:
            st.error("The UI encountered an unexpected error.")
            st.write(str(e))
            st.write(
                "**What you can do now:**\n"
                "1) Review your inputs and correct the bug if possible.\n"
                "2) Re-launch the UI.\n"
                "3) If the error keeps happening, contact the tech team at **emilien.caudron@pwc.com**."
            )
