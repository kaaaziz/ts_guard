DEFAULT_VALUES = {
    # Data collection timeout (Δt_i) in minutes.
    # If a sensor stays missing longer than this, we treat it as "timed out"
    # and trigger imputation + alerts
    "sigma_threshold": 10,
    # Graph Size
    "graph_size": 36,
    # System State Threshold
    "gauge_green_min": 0,
    "gauge_green_max": 20,
    "gauge_yellow_min": 20,
    "gauge_yellow_max": 50,
    "gauge_red_min": 50,
    "gauge_red_max": 100,
    # Trainig File path
    "training_file_path": "generated/model_TSGuard.pth",

    # 🕒 Simulation time scale: real seconds per simulated hour (0.0 = “max speed”)
    "sim_seconds_per_hour": 0.0,

    # 🔧 Constraint sensitivity: 0.0 = only large violations, 1.0 = any violation
    "constraint_sensitivity": 1.0,  
}

COLOR_MAP = {
    "active": "#90ee90",
    "inactive": "red",
    "imputed": "orange"
}