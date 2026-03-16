from __future__ import annotations

import html
import re

import streamlit as st
import streamlit.components.v1 as components

from integrations.llm.context import build_snapshot
from integrations.llm.service import generate_chat_reply


def _init_chat_state():
    ss = st.session_state
    if "chat_history" not in ss:
        ss["chat_history"] = []


def _format_message_html(text: str) -> str:
    safe = html.escape(str(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return safe


def render_chatbot():
    """
    Same UI as before.
    Only the backend 'brain' changes:
      - primary: Groq LLM
      - fallback: existing rule-based assistant
    """
    _init_chat_state()
    snap = build_snapshot()

    if "_chat_settings_css" not in st.session_state:
        st.session_state["_chat_settings_css"] = True

    st.markdown("### 🤖 TSGuard Assistant")
    st.caption(
        "Ask about the simulation status, missing / delayed values, "
        "or constraint / neighbour alerts."
    )

    messages_box = st.container(border=False)

    with messages_box:
        history = st.session_state["chat_history"]

        bubbles = []
        for msg in history:
            cls = (
                "tsguard-chat-msg-user"
                if msg["role"] == "user"
                else "tsguard-chat-msg-assistant"
            )
            safe_text = _format_message_html(msg["text"])
            bubbles.append(f"<div class='{cls}'>{safe_text}</div>")

        html_block = f"""
        <html>
          <head>
            <style>
              body {{
                margin: 0;
                padding: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 13px;
              }}
              .tsguard-chat-card {{
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                padding: 10px 12px;
              }}
              .tsguard-chat-body {{
                padding-right: 6px;
                margin-top: 4px;
                margin-bottom: 8px;
                max-height: 220px;
                overflow-y: auto;
              }}
              .tsguard-chat-msg-user {{
                background: #e0edff;
                padding: 6px 10px;
                border-radius: 10px;
                margin-bottom: 4px;
                text-align: left;
              }}
              .tsguard-chat-msg-assistant {{
                background: #f9fafb;
                padding: 6px 10px;
                border-radius: 10px;
                margin-bottom: 4px;
                text-align: left;
              }}
            </style>
          </head>
          <body>
            <div class="tsguard-chat-card">
              <div class="tsguard-chat-body" id="tsguard-chat-body">
                {''.join(bubbles)}
              </div>
            </div>
            <script>
              const el = document.getElementById('tsguard-chat-body');
              if (el) {{
                el.scrollTop = el.scrollHeight;
              }}
            </script>
          </body>
        </html>
        """

        components.html(html_block, height=260, scrolling=False)

    with st.form("tsguard_chat_form", clear_on_submit=True):
        user_msg = st.text_input(
            "Type a message…",
            key="tsguard_chat_input",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_msg.strip():
        text = user_msg.strip()

        reply = generate_chat_reply(
            user_text=text,
            snapshot=snap,
            chat_history=st.session_state["chat_history"],
        )

        st.session_state["chat_history"].append({"role": "user", "text": text})
        st.session_state["chat_history"].append({"role": "assistant", "text": reply})

        st.rerun()