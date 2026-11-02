# Economic assumptions
DOWNTIME_DAYS = 1

GLUCOSE_COST = 10.0        # ($/L) per mM of initial glucose
GLUTAMINE_COST = 20.0      # ($/L) per mM of initial glutamine
VCD_COST = 0.0005          # ($/L) per cell/mL of initial VCD
MAB_VALUE = 1000.0         # ($/L) per mg/L of mAb

HOURS_PER_DAY = 24.0

# Model and simulation settings
N_SIMULATIONS = 10
P_MAX = 4
CV_SPLITS = 3
POLY_DEGREE_MAX = 3
KNN_K_MAX = 5
RANDOM_SEED = 0

# Forecast settings
PREDICTION_INTERVAL = 0.95