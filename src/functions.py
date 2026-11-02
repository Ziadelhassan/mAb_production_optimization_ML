import sys
import numpy as np
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import src.constants as const

def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    # Store lagged predictors (x_lag), current values (x_cur), and corresponding times (t_cur)
    # from each batch separately before combining them.
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:
        # Select rows belonging to the current batch and extract all variables included in the VAR model.
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        # Current observations x(t).
        # Each row contains the simultaneous values of all variables at a particular time point.
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        # Build lagged state vectors:
        #     x(t-1), x(t-2), ..., x(t-p)
        # where each lag contains all process variables.
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lagged state.
            lagged_values.append(x_batch[p - i:-i])

        # Combine all lagged state vectors into a single feature matrix.
        # For example, with two variables and p = 3:
        #     [x1(t-1), x2(t-1), x1(t-2), x2(t-2), x1(t-3), x2(t-3)]
        # Each row therefore contains the lag history used to predict the current state.
        lagged_values = np.concatenate(lagged_values, axis=1)

        # Store this batch's feature matrix, target matrix, and time values.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    # Concatenate the independently generated batches into overall input and output arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur

def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")

def score_mab(estimator, x, y, idx_fp):
    y_pred = estimator.predict(x)
    return -root_mean_squared_error(y[:, idx_fp], y_pred[:, idx_fp])

def calculate_economics(source, model_type, p, batch_id,
        initial_glucose, initial_glutamine, initial_vcd,
        time_hours, mab, mab_lower=None, mab_upper=None):

    harvest_days = np.asarray(time_hours, dtype=float) / const.HOURS_PER_DAY
    mab = np.asarray(mab, dtype=float)

    culture_cost = (const.GLUCOSE_COST * initial_glucose
                    + const.GLUTAMINE_COST * initial_glutamine
                    + const.VCD_COST * initial_vcd)

    culture_value = const.MAB_VALUE * mab
    profit = culture_value - culture_cost
    cycle_days = harvest_days + const.DOWNTIME_DAYS

    # Exclude t=0: it is the initial state, not a production harvest
    valid = ((harvest_days > 0)
        & (cycle_days > 0)
        & np.isfinite(harvest_days)
        & np.isfinite(mab))

    df_economic = pd.DataFrame({
        "Source": source,
        "Model": model_type,
        "p": p,
        "Batch ID": batch_id,
        "Initial glucose [mM]": initial_glucose,
        "Initial glutamine [mM]": initial_glutamine,
        "Initial VCD [cells/mL]": initial_vcd,
        "Harvest time [d]": harvest_days[valid],
        "Downtime [d]": const.DOWNTIME_DAYS,
        "mAb [mg/L]": mab[valid],
        "Culture cost [$/L]": culture_cost,
        "Culture value [$/L]": culture_value[valid],
        "Profit [$/L]": profit[valid],
        "Space-time profit [$/(L d)]": (profit[valid] / cycle_days[valid]),
        "Space-time yield [mg/(L d)]": (mab[valid] / cycle_days[valid]),})

    # Convert the mAb prediction bounds into economic prediction bounds.
    for label, mab_bound in [("lower", mab_lower), ("upper", mab_upper)]:

        if mab_bound is None:
            # Reference data have no model-based prediction interval.
            df_economic[f"mAb PI {label} [mg/L]"] = np.nan
            df_economic[f"Culture value PI {label} [$/L]"] = np.nan
            df_economic[f"Space-time profit PI {label} [$/(L d)]"] = np.nan
            df_economic[f"Space-time yield PI {label} [mg/(L d)]"] = np.nan

        else:
            mab_bound = np.asarray(mab_bound, dtype=float)[valid]

            value_bound = const.MAB_VALUE * mab_bound
            profit_bound = value_bound - culture_cost

            df_economic[f"mAb PI {label} [mg/L]"] = mab_bound
            df_economic[f"Culture value PI {label} [$/L]"] = value_bound
            df_economic[f"Space-time profit PI {label} [$/(L d)]"] = (
                profit_bound / cycle_days[valid]
            )
            df_economic[f"Space-time yield PI {label} [mg/(L d)]"] = (
                mab_bound / cycle_days[valid]
            )

    return df_economic