
# AI4Semantics

AI4Semantics is a Streamlit-based application for interacting with LLMs and semantic models. It provides a chat interface, model upload/conversion utilities, and integration with MCP servers.

## Requirements

- Python 3.10 or above (ideally 3.12)

## Setup

1. Clone the repository: `git clone https://github.com/pwc-be-adv-tc-cd/AI4semantics`
2. Obtain the following files if not present: `.streamlit/`, `.openai_client_config.yaml`, `.env`
3. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `./.venv/Scripts/activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Repository Structure

```
├── app.py                  # Main Streamlit app entry point
├── chat_history/           # Conversation history and prompt templates
│   ├── chat_history.py     # ChatHistory class for saving/loading conversations
│   └── ...
├── chat_interface/         # Chat UI logic and data model utilities
│   ├── chat_interface.py   # Main chat interface logic
│   ├── chat_logic.py       # User input and chat message handling
│   └── data_model_utils/   # Utilities for model transformation/visualization
│       ├── chat_data_structure.py  # Transforms user data model for LLMs
│       ├── import_xml.py, export_xml.py, ...
├── clients/                # API clients for LLMs and MCP
│   ├── mcp_client.py       # MCP server client
│   ├── openai_client.py    # OpenAI API client
│   └── .openai_client_config.yaml  # OpenAI client config
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── ...
```

## Usage

1. Launch the app: `streamlit run app.py`
2. Set your user name and session in the sidebar:
   - Click `Set user` after entering your name
   - Set or reload a session to save/load conversations and models
3. Upload an XML model or let the app fetch it from the server
4. Use the chat interface to interact with the LLM and your model

## Folder Details

- **chat_history/**: Stores chat logs, prompt templates, and session histories
- **chat_interface/**: Contains all logic for the chat UI and model transformation utilities
  - `data_model_utils/`: Functions for importing/exporting/visualizing models and preparing data for LLMs
- **clients/**: Contains API clients for OpenAI and MCP, plus configuration

---

For more details, see comments in each module or contact the maintainers.
