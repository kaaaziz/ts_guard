import streamlit as st
from utils.config import DEFAULT_VALUES
import components.chatbot as chatbot 

import pandas as pd
import numpy as np

# ----------------------------
# Setting Management
# ----------------------------
def add_setting_panel():
    # Inject CSS once to make the *content* area of the settings expander scrollable.
    if "_settings_scroll_css" not in st.session_state:
        st.markdown(
            """
            <style>
              /* Make the vertical block that contains the settings tabs scrollable */
              div[data-testid="stVerticalBlock"]:has(> div#tsguard-settings-anchor) {
                max-height: 520px;
                overflow-y: auto;
                padding-right: 8px;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.session_state["_settings_scroll_css"] = True

    with st.expander("⚙️ Settings", expanded=True):
        # Anchor so the CSS above only affects this block
        st.markdown("<div id='tsguard-settings-anchor'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            [
                "📌 Constraints",
                "📈 Threshold",
                "📊 Missing values",
                "🕸️ Graph Options",
                "⏱️ Simulation",
                "📡 Captors",
                "📊 Models Comparison",
                "🤖 Assistant",
            ]
        )

        with tab1:
            add_constraints_panel()
            st.json(st.session_state.get("constraints", []))

        with tab2:
            add_threshold_panel()

        with tab3:
            add_missing_value_panel()

        with tab4:
            add_graph_opt_panel()

        with tab5:
            add_simulation_panel()

        with tab6:
            add_captor_panel()

        with tab7:
            add_models_comparison_panel()
        
        with tab8:
            chatbot.render_chatbot()


# ----------------------------
# Constraints Management
# ----------------------------
def add_constraints_panel():
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
    ctype = st.radio("Select Constraint Type", options=["📍 Spatial", "⏳ Temporal"], key="constraint_type")
    if "Spatial" in ctype:
        st.markdown("#### 📍 Spatial Constraints")
        # Distance with unit selection
        col1, col2 = st.columns([2, 1])
        with col1:
            spatial_distance = st.number_input("📏 Distance Threshold", value=2.0, step=0.1, key="spatial_distance")
        with col2:
            distance_unit = st.selectbox("Unit", ["km", "miles"], key="distance_unit")

        # Convert miles to km for standardization 
        spatial_distance_km = 0
        spatial_distance_miles = 0
        if distance_unit == "miles":
            spatial_distance_km = round(spatial_distance * 1.60934, 2)  # 1 mile = 1.60934 km
            spatial_distance_miles = spatial_distance
        else:
            spatial_distance_km = spatial_distance
            spatial_distance_miles = round(spatial_distance / 1.60934, 2)  # 1 mile = 1.60934 km

        spatial_diff = st.number_input("📊 Max Sensor Difference", value=5.0, step=0.1, key="spatial_diff")
        if st.button("Add Spatial Constraint", key="add_spatial"):
            st.session_state['constraints'].append({"type": "Spatial", "distance in km": spatial_distance_km, "distance in miles": spatial_distance_miles, "diff": spatial_diff})
            st.success("Spatial constraint added.")
    else:
        st.markdown("#### ⏳ Temporal Constraints")
        month = st.selectbox("🌦️ Month", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], key="month")
        constraint_option = st.selectbox("📉 Constraint Option", options=["Greater than", "Less than"], key="constraint_option")
        temp_threshold = st.number_input("📈 Threshold Value", value=50.0, step=0.1, key="temp_threshold")
        if st.button("Add Temporal Constraint", key="add_temporal"):
            st.session_state['constraints'].append({"type": "Temporal", "month": month, "option": constraint_option, "temp_threshold": temp_threshold})
            st.success("Temporal constraint added.")
# ----------------------------
# Missing value Management
# ----------------------------
def add_missing_value_panel():
    if 'missing_value_thresholds' not in st.session_state:
        st.session_state['missing_value_thresholds'] = []
    
    st.markdown("### 🛠 Define Missing Value Thresholds")
    st.markdown("Please specify the missing value percentage ranges for different risk states (Green: Low, Yellow: Medium, Red: High).")
    
    col1, col2 = st.columns(2)    
    with col1:
        green_min = st.number_input("🟢 Green Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_min"], step=1)
        yellow_min = st.number_input("🟡 Yellow Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_yellow_min"], step=1)
        red_min = st.number_input("🔴 Red Min", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_min"], step=1)
        
    with col2:
        green_max = st.number_input("🟢 Green Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_green_max"], step=1)
        yellow_max = st.number_input("🟡 Yellow Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_yellow_max"], step=1)
        red_max = st.number_input("🔴 Red Max", min_value=0, max_value=100, value=DEFAULT_VALUES["gauge_red_max"], step=1)
        
    if st.button("✅ Save Thresholds"):
        if not (green_min <= green_max <= yellow_min <= yellow_max <= red_min <= red_max):
            st.error("🚨 Invalid threshold ranges. Ensure consistency between min/max values.")
        else:
            st.session_state['missing_value_thresholds'] = {
                "Green": (green_min, green_max),
                "Yellow": (yellow_min, yellow_max),
                "Red": (red_min, red_max)
            }
            st.success("✅ Missing value thresholds saved successfully.")
        

# ----------------------------
# Threshold Management
# ----------------------------
def add_threshold_panel():
    if 'sigma_threshold' not in st.session_state:
        st.session_state['sigma_threshold'] = DEFAULT_VALUES["sigma_threshold"]
    
    st.markdown("Please specify the allowed delay threshold before a sensor is considered as having a missing value.")
    st.markdown("The default value is **" + str(DEFAULT_VALUES["sigma_threshold"]) + " minutes**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        threshold = st.number_input("📈 Threshold Value Delay", value=DEFAULT_VALUES["sigma_threshold"], step=1, key="threshold")
    with col2:
        time_unit = st.selectbox("Unit", ["minutes", "hours"], key="time_unit")
    '''if st.button("Set the delay threshold", key="set_sigma_threshold"):
        st.session_state['sigma_threshold'] = threshold
        st.success("Delay 'Sigma' threshold set to : **"+ str(threshold)+ " "+ time_unit+"**.")'''
    if st.button("Set the delay threshold", key="set_sigma_threshold"):
        # Internally we always store σ in *minutes*
        stored_val = threshold * 60 if time_unit == "hours" else threshold
        st.session_state['sigma_threshold'] = stored_val
        st.success(
            f"Delay 'Sigma' threshold set to : **{threshold} {time_unit}**."
        )

    st.markdown("---")
    st.markdown("### ⚖️ Constraint Sensitivity")
    st.markdown(
        "Controls how strict TSGuard is when raising alerts for constraint violations.  \n"
        "- Move **right** → more alerts (even small violations)  \n"
        "- Move **left** → fewer alerts (only large deviations)"
    )

    if 'constraint_sensitivity' not in st.session_state:
        st.session_state['constraint_sensitivity'] = DEFAULT_VALUES["constraint_sensitivity"]

    sensitivity = st.slider(
        "Constraint sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state['constraint_sensitivity']),
        step=0.05,
        help="0.0 = only large violations trigger an alert, 1.0 = any violation triggers.",
    )
    st.session_state['constraint_sensitivity'] = float(sensitivity)



# ----------------------------
# Graph Management
# ----------------------------
def add_graph_opt_panel():
    if 'graph_size' not in st.session_state:
        st.session_state['graph_size'] = DEFAULT_VALUES["graph_size"]
    
    st.markdown("### Configure Graph Size")  
    st.markdown("Specify the number of sensors (nodes) in the graph.")  
    st.markdown(f"**Default:** {DEFAULT_VALUES['graph_size']} sensors")

    g_size = st.number_input("📶 Graph Size", value=DEFAULT_VALUES["graph_size"], step=1, key="g_size")

    if st.button("Save", key="set_graph_size"):
        st.session_state['graph_size'] = g_size
        st.success("The graph size set to : **"+ str(g_size)+" sensors**.")


# ----------------------------
# Simulation Management
# ----------------------------
def add_simulation_panel():
    # 1) Initialise from defaults/session
    if "sim_seconds_per_hour" not in st.session_state:
        st.session_state["sim_seconds_per_hour"] = DEFAULT_VALUES.get("sim_seconds_per_hour", 0.0)

    current = float(st.session_state["sim_seconds_per_hour"])

    st.markdown("### ⏱️ Simulation Speed")
    st.markdown(
        "Control how fast the historical timeline is replayed during a simulation.\n\n"
        "- **0.0** → run as fast as possible (no delay)\n"
        "- **0.5** → 1 simulated hour ≈ 0.5 real seconds\n"
        "- **2.0** → 1 simulated hour ≈ 2 real seconds"
    )

    sim_seconds_per_hour = st.slider(
        "Real seconds per simulated hour",
        min_value=0.0,
        max_value=10.0,
        value=current,
        step=0.1,
        help=(
            "TSGuard replays your time series one timestamp at a time. "
            "This controls how many *real* seconds correspond to one "
            "*simulated* hour between two consecutive timestamps."
        ),
    )

    st.session_state["sim_seconds_per_hour"] = float(sim_seconds_per_hour)



# ----------------------------
# Dynamic captor management
# ----------------------------
def add_captor_panel():
    """
    UI for adding new sensors at runtime AND for forcing captors offline
    (hold-out mode).

    - st.session_state['dynamic_captors'] holds user-added captors.
    - st.session_state['forced_off_captors'] is a list of captor IDs that
      TSGuard should treat as "not sending values" (always missing) at runtime.
    """
    if "dynamic_captors" not in st.session_state:
        st.session_state["dynamic_captors"] = {}

    st.markdown("### 📡 Dynamic Captors & Hold-out Mode")

    # --- manual deactivation / hold-out captors ------------------------
    base_ids = [str(s) for s in st.session_state.get("sensor_list", [])]

    # Include dynamic captors as well (if any)
    dyn_ids = []
    for key, meta in st.session_state.get("dynamic_captors", {}).items():
        dyn_ids.append(str(meta.get("sensor_id", key)))

    options = sorted(set(base_ids + dyn_ids))

    # Keep only still-existing IDs in the default selection
    current_forced = [
        s for s in st.session_state.get("forced_off_captors", [])
        if s in options
    ]

    forced_off = st.multiselect(
        "Force these captors offline (hold-out mode)",
        options=options,
        default=current_forced,
        help=(
            "Selected captors keep their metadata and history, but TSGuard will "
            "ignore their raw values going forward and treat them as missing. "
            "The hybrid imputer reconstructs their values at each timestamp."
        ),
    )
    st.session_state["forced_off_captors"] = [str(s) for s in forced_off]

    st.markdown("---")

    # --- dynamic captor UI ---------------------------------------

    st.markdown(
        "Add new sensors while the simulation is running. "
        "New captors are immediately visible on the map and use the "
        "rule-based spatial imputer (no neural model) until the next training."
    )

    # Inject CSS ONCE to turn the file_uploader into a single "Add from file" button
    if "_captor_file_css" not in st.session_state:
        st.markdown(
            """
            <style>
            /* Scope all of this to our custom wrapper */
            #captor-file-wrapper [data-testid="stFileUploader"] {
                padding: 0;
            }

            /* Remove the big card / drag-and-drop area */
            #captor-file-wrapper [data-testid="stFileUploader"] section {
                padding: 0;
                border: none;
                background: transparent;
            }
            #captor-file-wrapper [data-testid="stFileUploader"] section > div:first-child {
                display: none;  /* hides "Drag and drop file here" + size text */
            }

            /* Hide any label text */
            #captor-file-wrapper label {
                display: none;
            }

            /* Make the internal button look like a single 'Add from file' button */
            #captor-file-wrapper [data-testid="stFileUploader"] button {
                width: 100%;
                justify-content: center;
                font-size: 0;            /* hide 'Browse files' text */
            }
            #captor-file-wrapper [data-testid="stFileUploader"] button:after {
                content: "Add from file";  /* this is what you see */
                font-size: 0.875rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_captor_file_css"] = True

    # ---- Manual add form with "Add captor" + "Add from file" side by side ----
    with st.form("add_dynamic_captor", clear_on_submit=True):
        new_id = st.text_input("Sensor ID", placeholder="e.g. 000037")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", format="%.6f")
        with c2:
            lon = st.number_input("Longitude", format="%.6f")

        col_manual, col_file = st.columns(2)
        with col_manual:
            submitted = st.form_submit_button("➕ Add captor")

        # This uploader is visually just a button ("Add from file")
        with col_file:
            st.markdown('<div id="captor-file-wrapper">', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Add from file",
                type=["csv", "txt"],
                accept_multiple_files=False,
                key="captor_file",
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- handle manual add ----
    if submitted:
        new_id = new_id.strip()
        if not new_id:
            st.warning("Please provide a sensor ID.")
        else:
            st.session_state["dynamic_captors"][new_id] = {
                "sensor_id": new_id,
                "latitude": lat,
                "longitude": lon,
            }
            st.success(f"Captor **{new_id}** added.")

    # ---- handle file selection: choosing a file immediately adds captors ----
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            required = {"sensor_id", "latitude", "longitude"}
            missing_cols = required - set(df.columns)
            if missing_cols:
                st.error(
                    "File is missing required columns: "
                    + ", ".join(sorted(missing_cols))
                )
            else:
                n_added = 0
                for _, row in df.iterrows():
                    sid = str(row["sensor_id"]).strip()
                    if not sid:
                        continue
                    try:
                        lat_val = float(row["latitude"])
                        lon_val = float(row["longitude"])
                    except (TypeError, ValueError):
                        continue

                    # overwrite or add
                    st.session_state["dynamic_captors"][sid] = {
                        "sensor_id": sid,
                        "latitude": lat_val,
                        "longitude": lon_val,
                    }
                    n_added += 1

                if n_added > 0:
                    st.success(f"Imported **{n_added}** captor(s) from file.")
                else:
                    st.warning("No valid captors were found in the file.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    # ---- show current dynamic captors ----
    if st.session_state["dynamic_captors"]:
        st.markdown("#### Active dynamic captors")
        st.json(st.session_state["dynamic_captors"])


# ----------------------------
# Models Comparison Management
# ----------------------------
def add_models_comparison_panel():
    """
    Live per-captor model comparison (TSGuard vs PriSTI, etc.).
    Reads a snapshot produced by run_simulation_with_live_imputation.
    """
    SS = st.session_state
    st.markdown("### 📊 Models Comparison")

    # ---- PriSTI debug: show last internal error if any ----
    pristi_err = SS.get("pristi_last_error")
    if pristi_err:
        st.warning(f"PriSTI internal error: {pristi_err}")
    # -------------------------------------------------------

    # 1) Which models are available?
    available_raw = SS.get("available_models", {"TSGuard"})
    if isinstance(available_raw, set):
        available = sorted(available_raw)
    elif isinstance(available_raw, (list, tuple)):
        available = sorted(set(available_raw))
    else:
        available = ["TSGuard"]

    if "TSGuard" not in available:
        available.insert(0, "TSGuard")

    # 2) Multiselect for models (TSGuard always enforced)
    default_models = SS.get("comparison_models", ["TSGuard"])
    default_valid = [m for m in default_models if m in available] or ["TSGuard"]

    selected = st.multiselect(
        "Select which models to display:",
        options=available,
        default=default_valid,
        help="TSGuard is always shown; PriSTI appears only if its artifacts are available.",
    )
    if "TSGuard" not in selected:
        selected.insert(0, "TSGuard")
    SS["comparison_models"] = selected

    col_left, col_right = st.columns([1, 2], gap="large")

    # 3) Captor list (all captors from dataset if possible)
    snapshot = SS.get("model_comparison_snapshot")
    captor_ids = SS.get("sensor_list")

    if not captor_ids and snapshot:
        ts_vals = snapshot.get("TSGuard_values") or {}
        captor_ids = list(ts_vals.keys())

    if not captor_ids:
        with col_left:
            st.info("Captors will appear here once data is loaded.")
        with col_right:
            st.info("Start TSGuard Simulation to see live model outputs.")
        return

    captor_ids = sorted({str(c) for c in captor_ids})

    # Left: captor selector
    with col_left:
        st.markdown("#### Captors")

        # Fixed-height scrollable box for the captor list
        with st.container(height=260, border=False):   # tweak 260 -> 300 if you want it taller
            current = SS.get("comparison_selected_captor")
            try:
                idx = captor_ids.index(current) if current in captor_ids else 0
            except ValueError:
                idx = 0

            selected_captor = st.radio(
                "Select a captor to inspect",
                options=captor_ids,
                index=idx,
                key="comparison_selected_captor",
            )



    # Right: details for that captor
    with col_right:
        if not snapshot:
            st.info("Start TSGuard Simulation to see live model outputs.")
            return

        ts = snapshot.get("timestamp")
        coords_by_col = SS.get("captor_coords_by_data_col") or {}
        coords_by_id = SS.get("captor_coords_by_sensor_id") or {}
        coord = coords_by_col.get(selected_captor) or coords_by_id.get(selected_captor)

        st.markdown(f"#### Captor {selected_captor}")
        if ts is not None:
            st.write(f"**Time:** {ts}")

        if coord:
            try:
                lat = float(coord.get("latitude"))
                lon = float(coord.get("longitude"))
                st.write(f"**Coordinates:** {lat:.4f}, {lon:.4f}")
            except Exception:
                st.write("**Coordinates:** —")
        else:
            st.write("**Coordinates:** —")

        # Build comparison table
        rows = []
        for model_name in selected:
            values_key = f"{model_name}_values"
            flags_key = f"{model_name}_imputed"
            vals = snapshot.get(values_key) or {}
            flags = snapshot.get(flags_key) or {}

            raw_val = vals.get(selected_captor)

            if isinstance(raw_val, (int, float, np.floating)):
                value_str = "N/A" if np.isnan(raw_val) else f"{float(raw_val):.3f}"
            elif raw_val is None:
                value_str = "N/A"
            else:
                try:
                    v = float(raw_val)
                    value_str = f"{v:.3f}"
                except Exception:
                    value_str = "N/A"

            status = "imputed" if bool(flags.get(selected_captor, False)) else "observed"

            if value_str == "N/A":
                status = "missing"

            rows.append(
                {
                    "Model": model_name,
                    "Value": value_str,
                    "Status": status,
                }
            )

        if rows:
            st.table(pd.DataFrame(rows))
        else:
            st.info("No model outputs available yet for this captor.")

        # Optional difference when two models are selected (first two)
        if len(selected) >= 2:
            m1, m2 = selected[:2]
            v1 = snapshot.get(f"{m1}_values", {}).get(selected_captor)
            v2 = snapshot.get(f"{m2}_values", {}).get(selected_captor)

            if all(
                isinstance(v, (int, float, np.floating)) and not np.isnan(v)
                for v in (v1, v2)
            ):
                diff = float(v2) - float(v1)
                st.markdown(f"**Δ({m2} – {m1}) = {diff:.3f}**")
