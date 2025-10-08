# ----- Imports
import pandas as pd
import streamlit as st
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T  # keep if you use types elsewhere

# Try to reuse an active session (works inside Snowflake). Otherwise, build one from secrets (for Streamlit Cloud).
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# ----- Streamlit UI config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title(":truck: Tasty Bytes Support: Customer Q&A Assistant  :truck:")
st.caption(
    "Welcome! This application suggests answers to customer questions based "
    "on corporate documentation and previous agent responses in support chats."
)

# ----- Get/Build Snowflake session
def get_session():
    try:
        return get_active_session()  # Works when running inside Snowflake
    except Exception:
        # Fallback for Streamlit Cloud (or local)
        cfg = st.secrets["snowflake"]
        return (
            Session.builder.configs(
                {
                    "account": cfg["account"],
                    "user": cfg["user"],
                    "password": cfg["password"],
                    "warehouse": cfg["warehouse"],
                    "database": cfg["database"],
                    "schema": cfg["schema"],
                    **({"role": cfg["role"]} if "role" in cfg else {}),
                }
            ).create()
        )

session = get_session()

# ----- Constants
CHAT_MEMORY = 20
DOC_TABLE = "app.vector_store"  # adjust if needed

# ----- Utilities
def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "What question do you need assistance answering?",
        }
    ]

# Use Cortex via SQL instead of Python SDK
def cortex_complete(model_name: str, prompt: str) -> str:
    # Use bind parameters to avoid quoting issues
    df = session.sql(
        "select snowflake.cortex.complete(?, ?) as RESPONSE",
        params=[model_name, prompt],
    ).to_pandas()
    return df["RESPONSE"].iloc[0]

def summarize(chat: str) -> str:
    # Keep the same instruction, but call SQL function
    prompt = (
        "Provide the most recent question with essential context "
        "from this support chat: " + chat
    )
    return cortex_complete(st.session_state.get("model", "mistral-large"), prompt)

def find_similar_doc(text: str, table_name: str) -> str:
    # Use EMBED_TEXT_768 via SQL and cosine similarity
    # Bind the user text; keep table name literal (identifiers can't be bound).
    q = f"""
        SELECT
            input_text,
            source_desc,
            VECTOR_COSINE_SIMILARITY(
                chunk_embedding,
                SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)
            ) AS dist
        FROM {table_name}
        ORDER BY dist DESC
        LIMIT 1
    """
    doc = session.sql(q, params=[text]).to_pandas()
    st.info("Selected Source: " + str(doc["SOURCE_DESC"].iloc[0]))
    return doc["INPUT_TEXT"].iloc[0]

def get_context(chat: str, table_name: str) -> str:
    chat_summary = summarize(chat)
    return find_similar_doc(chat_summary, table_name)

def get_prompt(chat: str, context: str) -> str:
    background = st.session_state.background_info
    prompt = f"""Answer this new customer question sent to our support agent
        at Tasty Bytes Food Truck Company. Use the background information
        and provided context taken from the most relevant corporate documents
        or previous support chat logs with other customers.
        Be concise and only answer the latest question.
        The question is in the chat.
        Chat: <chat> {chat} </chat>.
        Context: <context> {context} </context>.
        Background Info: <background_info> {background} </background_info>.
    """
    return prompt

# ----- Sidebar / settings
with st.expander(":gear: Settings"):
    st.session_state.model = st.selectbox(
        "Change chatbot model:",
        [
            "mistral-large",
            "reka-flash",
            "claude-4-sonnet",      # ensure your account has this exact alias; adjust if needed
            "llama4-maverick",      # ensure availability/alias
            "mistral-7b",
            "mixtral-8x7b",
            "snowflake-llama-3.1-405b",
        ],
        index=0,
    )
    st.button("Reset Chat", on_click=reset_conversation)

# ----- Load background doc once
if "background_info" not in st.session_state:
    st.session_state.background_info = (
        session.table("app.documents")
        .select("raw_text")
        .filter(F.col("relative_path") == "tasty_bytes_who_we_are.pdf")
        .collect()[0][0]
    )

# ----- Chat loop
if "messages" not in st.session_state:
    reset_conversation()

if user_message := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    chat = str(st.session_state.messages[-CHAT_MEMORY:])
    with st.chat_message("assistant"):
        with st.status("Answering..", expanded=True) as status:
            st.write("Finding relevant documents & support chat logs...")
            context = get_context(chat, DOC_TABLE)
            st.write("Using search results to answer your question...")
            prompt = get_prompt(chat, context)
            response = cortex_complete(st.session_state.model, prompt)
            status.update(label="Complete!", state="complete", expanded=False)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
