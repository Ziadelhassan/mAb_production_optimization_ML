import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import src.constants as const
from src.classes import StandardScalerVAR
from src.functions import (generate_x_lag_cur_var,progress_bar,score_mab,calculate_economics,)

# ============================================================
# CONFIGURE MODELS AND FORECAST SETTINGS
# ============================================================

n_simulations = const.N_SIMULATIONS
p_max = const.P_MAX
cv_splits = const.CV_SPLITS
poly_degree_max = const.POLY_DEGREE_MAX
knn_k_max = const.KNN_K_MAX

rng = np.random.default_rng(const.RANDOM_SEED)

prediction_interval = const.PREDICTION_INTERVAL
alpha = 1 - prediction_interval

#Non-linear models used with their GridSearch parameters
model_types = ["poly", "knn"]
param_grid_poly = dict(polynomialfeatures__degree = [i for i in range(2, poly_degree_max + 1)])   #to try different values. Shortcut pour:  for i in range(2,4): degrees.append(i)
param_grid_knn = dict(kneighborsregressor__n_neighbors = [i for i in range(2, knn_k_max + 1)],kneighborsregressor__weights = ["uniform"])


# ============================================================
# LOAD DATA AND PREPARE BATCHES AND INITIAL CONDITIONS
# ============================================================

batch_data = []
batch_files = sorted(filename for filename in os.listdir("datasets")
    if filename.startswith("batch_") and filename.endswith(".csv"))
for filename in batch_files:
    batch_id = int(filename.split("_")[1].split(".")[0])
    df_batch = pd.read_csv(os.path.join("datasets", filename))
    df_batch.insert(0, "Batch ID", batch_id)
    batch_data.append(df_batch)
df = pd.concat(batch_data, ignore_index=True)
df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))

# Extract the selected batch IDs and process variables
batch_ids = np.unique(df.loc[:, "Batch ID"])
features = df.columns[2:]

# Keep only the initial conditions corresponding to our selected batches
df_ic = df_ic.loc[df_ic["Batch ID"].isin(batch_ids)].copy()

# Select the variable that will be plotted and evaluated
feature_plot = "mAb [mg/L]"
units = feature_plot.split(" [")[-1][:-1]
idx_fp = np.argmax(features == feature_plot)

# Organize subplots according to initial conditions
ic_glucose = np.unique(df_ic.loc[:, "Glucose [mM]"])
ic_glutamine = np.unique(df_ic.loc[:, "Glutamine [mM]"])
ic_vcd = np.unique(df_ic.loc[:, "VCD [cells/mL]"])
nrows = ic_glucose.shape[0]
ncols = ic_glutamine.shape[0]


# ============================================================
# TRAIN AND EVALUATE MODELS USING LEAVE-ONE-BATCH-OUT CV
# ============================================================

n_iters = len(model_types) * p_max * batch_ids.shape[0] * n_simulations  # 2 * 4 * 30 * 10 = 2400
c_iter = 0
records = []
plot_data = {}

for model_type in model_types:
    for p in range(1, p_max + 1):

        plot_data[(model_type, p)] = []
        rmse_forecasts = []
        for batch_id_test in batch_ids:
            # ------ Prepare training data -----
            batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(p=p, batch_ids=batch_ids_train, df=df, features=features)

            a_min = np.min(x_cur_train, axis=0)  #used later to limit error
            a_max = np.max(x_cur_train, axis=0)

            # ------- Select hyperparameters and fit the model -----
            if model_type == "poly":
                model = make_pipeline(StandardScalerVAR(p=p), PolynomialFeatures(include_bias=False), LinearRegression())
                param_grid = param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(StandardScalerVAR(p=p), KNeighborsRegressor())
                param_grid = param_grid_knn
            else:
                model = None
                param_grid = {}

            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring=partial(score_mab, idx_fp=idx_fp),return_train_score=True)
            model_gscv.fit(x_lag_train, x_cur_train, groups=b_cur_train)

            # ------- Calculate training residuals -----
            x_cur_train_pred = model_gscv.predict(x_lag_train)
            x_cur_train_res = x_cur_train - x_cur_train_pred

            # ------- Prepare the held-out test batch -----
            idx_ic = (df_ic.loc[:, "Batch ID"] == batch_id_test).idxmax()
            initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
            initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
            initial_vcd = df_ic.at[idx_ic, "VCD [cells/mL]"]

            # Construct the initial state
            initial_state = {
                "Glucose [mM]": initial_glucose,
                "Glutamine [mM]": initial_glutamine,
                "Lactate [mM]": 0,
                "Ammonia [mM]": 0,
                "VCD [cells/mL]": initial_vcd,
                "TCD [cells/mL]": initial_vcd,
                "mAb [mg/L]": 0 }
            x_initial = np.array([initial_state[feature] for feature in features])

            # ------- Extract the data for the test batch -----
            mask = df.loc[:, "Batch ID"] == batch_id_test
            t_test = df.loc[mask, "Time [h]"].to_numpy()
            x_test = df.loc[mask, features].to_numpy()

            # ------- Simulate forecasts with residual noise -----
            x_forecasts = []
            for _ in range(n_simulations):
                # Initialize the forecast using the known initial conditions
                x_forecast = [x_initial]

                # Generate forecasts recursively
                while len(x_forecast) < t_test.shape[0]:
                    x_lag = x_forecast[::-1][:p].copy()
                    while len(x_lag) < p:
                        x_lag.append(x_lag[-1])

                    # Combine the lagged states into one input row
                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    # Sample training residuals for each of the seven variables
                    # This assumes the residuals behave like white noise
                    white_noise = []
                    for j in range(x_cur_train_res.shape[1]):
                        white_noise.append(rng.choice(x_cur_train_res[:, j]))
                    white_noise = np.array([white_noise])

                    # Predict the next state for all seven variables
                    x_cur_baseline = model_gscv.predict(x_lag)
                    x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max) #to Limit each variable to its own training range

                    x_cur = x_cur_baseline + white_noise

                    # Prevent negative concentrations and cell densities
                    x_cur = np.clip(x_cur[0], a_min=0, a_max=None)
                    x_forecast.append(x_cur)

                x_forecast = np.array(x_forecast)
                x_forecasts.append(x_forecast)

                c_iter += 1
                progress_bar(c_iter, n_iters)

            x_forecasts = np.array(x_forecasts)

            # ------- Evaluate forecasts and store plotting data -----
            x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
            x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
            x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

            # Calculate RMSE for mAb
            rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
            rmse_forecasts.append(rmse_forecast)

            message = f"Batch {batch_id_test} | RMSE: {rmse_forecast:.3f} {units}"

            # Extract the best hyperparameters
            if model_type == "poly":
                best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
                message += f" | Degree: {best_degree}"

            elif model_type == "knn":
                best_n_neighbors = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
                best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
                message += f" | k={best_n_neighbors}, {best_weight}"

            plot_data[(model_type, p)].append({
                "batch_id": batch_id_test,
                "initial_glucose": initial_glucose,
                "initial_glutamine": initial_glutamine,
                "initial_vcd": initial_vcd,
                "t_test": t_test.copy(),
                "measured": x_test[:, idx_fp].copy(),
                "median": x_forecast_median[:, idx_fp].copy(),
                "lower": x_forecast_pi_lower[:, idx_fp].copy(),
                "upper": x_forecast_pi_higher[:, idx_fp].copy(),
                "message": message,})

        # Summarize mAb forecast errors across the test batches
        rmse_mean = np.mean(rmse_forecasts)
        rmse_sd = np.std(rmse_forecasts)

        record = (model_type, p, rmse_mean, rmse_sd)
        records.append(record)

        # Print a summary after each model and p finishes
        print()
        print(f"Model: {model_type.upper()} | p={p} | "
              f"Forecast RMSE: {rmse_mean:.3f} +/- {rmse_sd:.3f} {units}")


# ============================================================
# COMPARE MODEL PERFORMANCE AND SELECT THE BEST MODELS
# ============================================================

columns = ["Model",
            "p",
            "Mean RMSE [mg/L]",
            "SD RMSE [mg/L]"]

df_results = pd.DataFrame.from_records(records, columns=columns)
df_results = df_results.sort_values("Mean RMSE [mg/L]")
best_result = df_results.iloc[0]
best_by_type = (df_results.sort_values(["Mean RMSE [mg/L]", "p"]).drop_duplicates(subset="Model"))

print()
print("MODEL COMPARISON: mAb FORECASTS")
print(df_results.to_string(index=False, float_format="{:.3f}".format))
print()
print(f"Lowest mean forecast RMSE: {best_result['Mean RMSE [mg/L]']:.3f} "
      f"+/- {best_result['SD RMSE [mg/L]']:.3f} mg/L")
print(f"   Model: {best_result['Model']}")
print(f"   p: {best_result['p']}")
print()
print("MODELS SELECTED FOR GRAPHS")
print(best_by_type.to_string(index=False, float_format="{:.3f}".format))

os.makedirs("tables", exist_ok=True)
df_results.to_csv(os.path.join("tables", "model_comparison.csv"), index=False)


# ============================================================
# CALCULATE ECONOMICS AND FIND THE BEST OPERATING CONDITIONS
# ============================================================

economic_tables = []
for _, result in best_by_type.iterrows():
    model_type = result["Model"]
    p = int(result["p"])

    for batch in plot_data[(model_type, p)]:
        economic_tables.append(calculate_economics(
            source="Forecast median",
            model_type=model_type,
            p=p,
            batch_id=batch["batch_id"],
            initial_glucose=batch["initial_glucose"],
            initial_glutamine=batch["initial_glutamine"],
            initial_vcd=batch["initial_vcd"],
            time_hours=batch["t_test"],
            mab=batch["median"],
            mab_lower=batch["lower"],
            mab_upper=batch["upper"],))

# Evaluate measured data once as an experimental comparison
for batch_id in batch_ids:
    initial = df_ic.loc[df_ic["Batch ID"] == batch_id].iloc[0]
    measured = df.loc[df["Batch ID"] == batch_id]

    economic_tables.append(calculate_economics(
        source="Measured",
        model_type="Experimental",
        p=0,
        batch_id=batch_id,
        initial_glucose=initial["Glucose [mM]"],
        initial_glutamine=initial["Glutamine [mM]"],
        initial_vcd=initial["VCD [cells/mL]"],
        time_hours=measured["Time [h]"].to_numpy(),
        mab=measured[feature_plot].to_numpy(),))

df_economics = pd.concat(economic_tables, ignore_index=True)

if df_economics.empty:
    raise ValueError("No valid harvest times available for economic analysis.")

profit_column = "Space-time profit [$/(L d)]"

df_economics = df_economics.sort_values(profit_column, ascending=False)
df_best_per_batch = df_economics.drop_duplicates(subset=["Source", "Model", "p", "Batch ID"])
df_economic_optima = df_best_per_batch.drop_duplicates(subset=["Source", "Model", "p"])

columns_economic = [
    "Source",
    "Model",
    "p",
    "Batch ID",
    "Initial glucose [mM]",
    "Initial glutamine [mM]",
    "Initial VCD [cells/mL]",
    "Harvest time [d]",
    "mAb [mg/L]",
    "Culture cost [$/L]",
    "Profit [$/L]",
    profit_column,
    "Space-time yield [mg/(L d)]",
    "Culture value PI lower [$/L]",
    "Culture value PI upper [$/L]",
    "Space-time profit PI lower [$/(L d)]",
    "Space-time profit PI upper [$/(L d)]",
    "Space-time yield PI lower [mg/(L d)]",
    "Space-time yield PI upper [mg/(L d)]",]

print()
print("ECONOMIC OPTIMA — selected initial conditions and sampled times")
print(df_economic_optima[columns_economic].to_string(index=False, float_format="{:.3f}".format))

# Use forecast accuracy to choose which model's recommendation to report
recommended_model = best_result["Model"]
recommended_p = int(best_result["p"])

recommended = df_economic_optima.loc[
    (df_economic_optima["Source"] == "Forecast median")
    & (df_economic_optima["Model"] == recommended_model)
    & (df_economic_optima["p"] == recommended_p)].iloc[0]

print()
print("RECOMMENDATION FROM THE LOWEST-RMSE MODEL")
print(f"Model: {recommended_model.upper()} | p={recommended_p}")
print(f"Initial glucose: {recommended['Initial glucose [mM]']:.1f} mM")
print(f"Initial glutamine: {recommended['Initial glutamine [mM]']:.1f} mM")
print(f"Initial VCD: {recommended['Initial VCD [cells/mL]']:,.0f} cells/mL")
print(f"Harvest time: {recommended['Harvest time [d]']:.1f} days")
print(f"Predicted mAb: {recommended['mAb [mg/L]']:.3f} mg/L")
print(f"Predicted space-time profit: "
    f"{recommended[profit_column]:.2f} $/(L d)")
print()
print("ECONOMIC ESTIMATES AT THE RECOMMENDED HARVEST TIME")

print(f"Culture cost: "
    f"{recommended['Culture cost [$/L]']:.2f} $/L")

print(f"Culture value: "
    f"{recommended['Culture value [$/L]']:.2f} $/L "
    f"(95% PI: "
    f"{recommended['Culture value PI lower [$/L]']:.2f} to "
    f"{recommended['Culture value PI upper [$/L]']:.2f})")

print(f"Space-time profit: "
    f"{recommended[profit_column]:.2f} $/(L d) "
    f"(95% PI: "
    f"{recommended['Space-time profit PI lower [$/(L d)]']:.2f} to "
    f"{recommended['Space-time profit PI upper [$/(L d)]']:.2f})")

print(f"Space-time yield: "
    f"{recommended['Space-time yield [mg/(L d)]']:.4f} mg/(L d) "
    f"(95% PI: "
    f"{recommended['Space-time yield PI lower [mg/(L d)]']:.4f} to "
    f"{recommended['Space-time yield PI upper [mg/(L d)]']:.4f})")

if recommended[profit_column] <= 0:
    print("No positive-profit option was predicted within this search.")

df_economics.to_csv(os.path.join("tables", "economics_all_harvest_times.csv"),index=False)
df_best_per_batch.to_csv(os.path.join("tables", "economics_best_harvest_per_batch.csv"),index=False)
df_economic_optima.to_csv(os.path.join("tables", "economics_optima.csv"),index=False)


# ============================================================
# PLOT AND SAVE FORECASTS FOR THE BEST MODELS
# ============================================================

os.makedirs("figures", exist_ok=True)

for _, result in best_by_type.iterrows():
    model_type = result["Model"]
    p = int(result["p"])

    batches_to_plot = plot_data[(model_type, p)]

    y_min = 0
    y_max = max(
        df[feature_plot].max(),
        max(batch["upper"].max() for batch in batches_to_plot))
    y_buffer = 0.05 * (y_max - y_min)

    for initial_vcd in ic_vcd:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(40.64 / 2.54, 25.4 / 2.54),
            dpi=200,
            layout="constrained",
            squeeze=False)

        fig.suptitle(
            f"BEST {model_type.upper()} NVAR({p}) | "
            f"Initial VCD: {initial_vcd:,.0f} cells/mL",
            fontsize=17)

        for batch in batches_to_plot:
            if batch["initial_vcd"] != initial_vcd:
                continue

            i_ax = np.argmax(ic_glucose == batch["initial_glucose"])
            j_ax = np.argmax(ic_glutamine == batch["initial_glutamine"])
            ax = axes[i_ax, j_ax]

            ax.scatter(
                batch["t_test"], batch["measured"],
                label="Test", s=10, color="tab:orange",
                edgecolors="black", zorder=3)
            ax.plot(
                batch["t_test"], batch["median"],
                label=f"NVAR({p})", color="black", zorder=4)
            ax.fill_between(
                batch["t_test"], batch["lower"], batch["upper"],
                label=f"{prediction_interval:.0%} PI",
                color="tab:blue", edgecolors="none",
                alpha=0.3, zorder=2)

            text_position = 0.04 + 0.10 * len(ax.texts)
            ax.text(
                x=0.03, y=text_position,
                s=batch["message"],
                va="bottom", ha="left",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(
                    facecolor="white", edgecolor="none", alpha=0.85) )

        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]

                ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
                ax.set_xlim(0, df["Time [h]"].max())

                ax.set_title(
                    f"Initial glucose: {ic_glucose[i]:.0f} mM | "
                    f"Initial glutamine: {ic_glutamine[j]:.0f} mM",
                    fontsize=11, pad=8)

                ax.xaxis.set_major_locator(MultipleLocator(168))
                ax.tick_params(axis="both", labelsize=10)

                if i == nrows - 1:
                    ax.set_xlabel("Time [h]", fontsize=12)
                else:
                    ax.tick_params(axis="x", labelbottom=False)

                if j == 0:
                    ax.set_ylabel(feature_plot, fontsize=12)
                else:
                    ax.tick_params(axis="y", labelleft=False)

                ax.grid(alpha=0.2)

                if i == 0 and j == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(handles[:3], labels[:3],loc="upper right", fontsize=9)

        fig.savefig(os.path.join(
            "figures",
            f"forecast_BEST_{model_type}_p{p:02d}_"
            f"VCD{initial_vcd:.0f}_N{n_simulations:06d}.png"))
        plt.close(fig)













