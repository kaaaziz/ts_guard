import streamlit as st

# ----------------------------
# Buttons Management
# ----------------------------
def add_buttons():
    if "page" not in st.session_state:
        st.session_state.page = "sim"   # "sim" or "cmp"

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🧠 Start TSGuard training", use_container_width=True):
            st.session_state.training = True

    with col2:
        if st.button("▶️ Start TSGuard Simulation", use_container_width=True):
            st.session_state.page = "sim"
            st.session_state.running = True

            # Force a fresh run namespace on every new experiment run
            st.session_state.pop("current_run_id", None)