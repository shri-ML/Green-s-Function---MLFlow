"""
FEM vs Green's function stress predictions using the paper's own
temperature-interpolation method (Rouse & Hyde, 2016) - NON-UNIFORM case.

Same method as "INTERPOLATOIN METHOD PAPER", only the inputs differ:

  green_non uniform.xlsx                       - unit-step Green's functions,
                                                 T_rep = 100..600 C, t = 0.01..100 s
  transient_non uniform/convection loading.xls - ambient temperature history
                                                 driving the transient
  transient_non uniform/transient_non uniform.xls
                                               - FEA (ANSYS) truth, von Mises
                                                 stress (unit read from header)

  Eq. 10: G(t) = exp(sum_m C_m * ln(t)^(m-1))
  Eq. 14: evaluate every fitted reference Green's function at the same lag.
  Eq. 15: sigma(T) = sum_i D_i * T^(i-1), interpolating polynomial in T,
          evaluated at the instantaneous temperature.

Note: the convection loading starts and ends at 22 C, below the lowest
reference temperature (100 C). Below 100 C the 100 C Green's function is
used (temperature is clamped to the reference range).
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
green_file = os.path.join(project_dir, "green_non uniform.xlsx")
loading_file = os.path.join(project_dir, "transient_non uniform", "convection loading.xls")
transient_file = os.path.join(project_dir, "transient_non uniform", "transient_non uniform.xls")
output_dir = os.path.join(script_dir, "results")
os.makedirs(output_dir, exist_ok=True)
comparison_output = os.path.join(output_dir, "FEM_vs_PaperInterpolation_non_uniform.csv")

PAPER_INTERP_MODE = "local"   # "global" = literal Eq.15 (all M points at once)
LOCAL_WINDOW = 4             # reference temperatures used per fit in "local" mode

# ANSYS text exports: stress unit is given in the column header.
UNIT_TO_MPA = {"[MPa]": 1.0, "[Pa]": 1e-6, "[hbar]": 10.0}

# ---------------------------------------------------------------------------
# Load the Green's function reference data: one unit-step stress history
# per reference temperature.
# ---------------------------------------------------------------------------
df = pd.read_excel(green_file)

T_col = "InitialTemperature_C"
time_col = "Time_s"
stress_col = "VonMises_MPa"

df[T_col] = pd.to_numeric(df[T_col])
df[time_col] = pd.to_numeric(df[time_col])
df[stress_col] = pd.to_numeric(df[stress_col])

temperatures = np.sort(df[T_col].unique()).astype(float)
green_times = np.sort(df[time_col].unique())

print("\nGreen's function reference data")
print(f"  M = {len(temperatures)} reference temperatures: {temperatures}")
print(f"  local window = {LOCAL_WINDOW} neighbouring reference temperatures per fit")
print(f"  time range:   {green_times[0]} to {green_times[-1]} s")


# ---------------------------------------------------------------------------
# Fit Eq. 10 (a 7-term exponential Green's function) to each reference
# temperature's unit-step curve.
# ---------------------------------------------------------------------------
def gf_model(C, t):
    """Eq. 10: G(t) = exp(sum_m C_m * ln(t)^(m-1))"""
    lnt = np.log(t)
    powers = np.vstack([lnt ** m for m in range(7)])
    return np.exp(C @ powers)


def fit_gf(t, s):
    """Levenberg-Marquardt fit of Eq. 10 to one temperature's data."""
    C0 = np.array([-0.2, 0.4, -0.02, -3e-4, -1e-4, -2e-4, 0.0])
    C0[0] += np.log(max(abs(s).max(), 1e-8))
    result = least_squares(lambda C: gf_model(C, t) - s, C0, method="lm", max_nfev=20000)
    pred = gf_model(result.x, t)
    r2 = 1 - np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2)
    return result.x, r2


print("\nFitting Eq. 10 per reference temperature")
coeffs_by_T = {}
r2_by_T = {}
lag_domain = (green_times[green_times > 0].min(), green_times.max())

print(f"{'T (C)':>8} | {'R2':>8}")
for T in temperatures:
    sub = df[df[T_col] == T].sort_values(time_col)
    t = sub[time_col].values.astype(float)
    s = sub[stress_col].values.astype(float)
    mask = t > 0  # ln(0) is undefined, so skip t=0
    C, r2 = fit_gf(t[mask], s[mask])
    coeffs_by_T[T] = C
    r2_by_T[T] = r2
    print(f"{T:8.1f} | {r2:8.5f}")

pd.DataFrame(
    [[T, *coeffs_by_T[T], r2_by_T[T]] for T in temperatures],
    columns=["T_C", *[f"C{m}" for m in range(7)], "R2"],
).to_csv(os.path.join(output_dir, "eq10_coefficients_by_temperature.csv"), index=False)


def G_eq10(T, lag):
    """Fitted Eq.10 Green's function at reference temperature T."""
    lag = np.clip(np.asarray(lag, dtype=float), *lag_domain)
    return gf_model(coeffs_by_T[T], lag)


# ---------------------------------------------------------------------------
# Eq. 14/15 - evaluate every reference Green's function at the current
# lag, then fit an exact interpolating polynomial through those
# (T, stress) points and evaluate it at the actual instantaneous
# temperature.
# ---------------------------------------------------------------------------
def local_window(T_query, window=LOCAL_WINDOW):
    """Indices of the `window` reference temperatures nearest T_query."""
    idx = np.argsort(np.abs(temperatures - T_query))[:window]
    return np.sort(idx)


# Precompute a (lag, T) surface once, then use fast bilinear lookup inside
# the convolution loop - refitting the interpolating polynomial at every
# convolution step would be far too slow.
print(f"\nBuilding Eq.14/15 surface ({PAPER_INTERP_MODE} mode)...")
lag_axis = np.geomspace(lag_domain[0], lag_domain[1], 300)
T_fine = np.linspace(temperatures.min(), temperatures.max(), 200)
G_paper_grid = np.zeros((len(lag_axis), len(T_fine)))

for i, lag in enumerate(lag_axis):
    sigma_refs = np.array([G_eq10(T, [lag])[0] for T in temperatures])
    if PAPER_INTERP_MODE == "global":
        poly = np.polynomial.polynomial.Polynomial.fit(temperatures, sigma_refs, deg=len(temperatures) - 1)
        G_paper_grid[i, :] = poly(T_fine)
    else:
        for j, Tq in enumerate(T_fine):
            idx = local_window(Tq)
            poly = np.polynomial.polynomial.Polynomial.fit(temperatures[idx], sigma_refs[idx], deg=len(idx) - 1)
            G_paper_grid[i, j] = poly(Tq)

lag_axis_full = np.insert(lag_axis, 0, 0.0)  # G(lag=0) = 0
G_paper_grid_full = np.vstack([np.zeros(len(T_fine)), G_paper_grid])
G_paper = RegularGridInterpolator(
    (lag_axis_full, T_fine), G_paper_grid_full, method="linear", bounds_error=False, fill_value=None)


def G_function_paper(points):
    pts = np.array(points, dtype=float)
    pts[:, 0] = np.clip(pts[:, 0], lag_axis_full.min(), lag_axis_full.max())
    pts[:, 1] = np.clip(pts[:, 1], T_fine.min(), T_fine.max())
    return G_paper(pts)


# ---------------------------------------------------------------------------
# Test transient: ambient temperature history from the ANSYS convection
# loading table (linear between breakpoints).
# ---------------------------------------------------------------------------
loading = pd.read_csv(loading_file, sep="\t", encoding="utf-8-sig")
load_time = pd.to_numeric(loading["Time [s]"], errors="coerce")
load_temp = pd.to_numeric(loading["Temperature [°C]"], errors="coerce")
ok = load_time.notna() & load_temp.notna()
load_time, load_temp = load_time[ok].to_numpy(), load_temp[ok].to_numpy()

print("\nConvection loading (ambient temperature history)")
for tb, Tb in zip(load_time, load_temp):
    print(f"  t={tb:6.1f} s  T={Tb:6.1f} C")

total_time, dt = float(load_time[-1]), 0.1

time = np.arange(0.0, total_time + 0.5 * dt, dt)
time[-1] = total_time
temperature = np.interp(time, load_time, load_temp)
dT = np.diff(temperature, prepend=temperature[0])

if temperature.min() < temperatures[0] or temperature.max() > temperatures[-1]:
    print(f"  WARNING: transient spans {temperature.min():.1f}-{temperature.max():.1f} C, outside the "
          f"reference range {temperatures[0]:.0f}-{temperatures[-1]:.0f} C; the nearest reference "
          f"Green's function is used outside it.")
if total_time > green_times[-1]:
    raise ValueError(f"Transient needs {total_time:.2f} s but Green's function only covers {green_times[-1]:.2f} s.")


# ---------------------------------------------------------------------------
# Convolve: sum G(t - tau) * dT(tau) over the transient's history, looking
# up G at the temperature the component was actually at, at each tau.
# ---------------------------------------------------------------------------
def calculate_convolution(G_function, name):
    print(f"\nCalculating: {name}")
    stress = np.zeros_like(time)
    for n in range(len(time)):
        tau = time[:n + 1]
        lag = time[n] - tau
        points = np.column_stack([lag, temperature[:n + 1]])
        stress[n] = np.sum(G_function(points) * dT[:n + 1])
    return stress


sigma_paper = calculate_convolution(G_function_paper, f"paper interpolation ({PAPER_INTERP_MODE})")


# ---------------------------------------------------------------------------
# Load the real FEA transient result to compare against.
# ---------------------------------------------------------------------------
print("\nReading FEA transient result")
transient = pd.read_csv(transient_file, sep="\t", encoding="utf-8-sig")
stress_header = transient.columns[2]
unit = next(u for u in UNIT_TO_MPA if stress_header.endswith(u))

fem_time = pd.to_numeric(transient.iloc[:, 1], errors="coerce").to_numpy()
fem_stress = pd.to_numeric(transient.iloc[:, 2], errors="coerce").to_numpy() * UNIT_TO_MPA[unit]
valid = np.isfinite(fem_time) & np.isfinite(fem_stress)
fem_time, fem_stress = fem_time[valid], fem_stress[valid]
order = np.argsort(fem_time)
fem_time, fem_stress = fem_time[order], fem_stress[order]

print(f"  stress column: '{stress_header}' -> converted {unit} to MPa (x{UNIT_TO_MPA[unit]:g})")
print(f"  {len(fem_time)} points, {fem_time[0]} to {fem_time[-1]} s")
print(f"  peak: {fem_stress.max():.2f} MPa at t={fem_time[np.argmax(fem_stress)]:.2f}s")


# ---------------------------------------------------------------------------
# Compare predictions to FEA over their common time range.
# ---------------------------------------------------------------------------
t_start = max(fem_time[0], time[0])
t_end = min(fem_time[-1], time[-1])
mask = (fem_time >= t_start) & (fem_time <= t_end)
comparison_time = fem_time[mask]
fem_compare = fem_stress[mask]

paper_compare = np.interp(comparison_time, time, sigma_paper)


def calculate_metrics(reference, prediction):
    error = reference - prediction
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    r2 = r2_score(reference, prediction)
    relative_rmse = rmse / np.max(np.abs(reference)) * 100
    return error, rmse, mae, max_error, relative_rmse, r2


error_paper, rmse_paper, mae_paper, max_paper, rel_paper, r2_paper = calculate_metrics(fem_compare, paper_compare)

print("\nFEM vs paper interpolation (Eq.14/15)")
print(f"{'method':<20}{'RMSE':>12}{'MAE':>12}{'max err':>12}{'R2':>10}")
print(f"{'paper interp':<20}{rmse_paper:12.4f}{mae_paper:12.4f}{max_paper:12.4f}{r2_paper:10.5f}")
print(f"relative RMSE: {rel_paper:.3f}%")
print(f"predicted peak: {sigma_paper.max():.2f} MPa at t={time[np.argmax(sigma_paper)]:.2f}s")

pd.DataFrame({
    "Time_s": comparison_time,
    "Temperature_C": np.interp(comparison_time, time, temperature),
    "FEM_VonMises_MPa": fem_compare,
    f"PaperInterp_{PAPER_INTERP_MODE}_MPa": paper_compare,
    "Error_MPa": error_paper,
}).to_csv(comparison_output, index=False)
print(f"\nSaved: {comparison_output}")

pd.DataFrame([{
    "method": f"paper interpolation ({PAPER_INTERP_MODE}, window={LOCAL_WINDOW})",
    "RMSE_MPa": rmse_paper, "MAE_MPa": mae_paper, "MaxAbsError_MPa": max_paper,
    "RelativeRMSE_pct": rel_paper, "R2": r2_paper,
    "FEM_peak_MPa": fem_stress.max(), "FEM_peak_time_s": fem_time[np.argmax(fem_stress)],
    "Pred_peak_MPa": sigma_paper.max(), "Pred_peak_time_s": time[np.argmax(sigma_paper)],
}]).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(fem_time, fem_stress, label="FEM", linewidth=2)
ax.plot(time, sigma_paper, label=f"paper interpolation ({PAPER_INTERP_MODE}, Eq.14/15)", linewidth=1.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Von Mises Stress [MPa]")
ax.set_title("Non-uniform case: FEM vs paper interpolation method (Eq. 9/10/14/15)")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fem_vs_paper_interpolation.png"), dpi=150)

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(comparison_time, error_paper, label="paper interpolation error", linewidth=1.5)
ax2.axhline(0, linestyle="--", color="k")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("FEM - prediction [MPa]")
ax2.set_title("Non-uniform case: prediction error")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "paper_interpolation_error.png"), dpi=150)

fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(time, temperature, linewidth=2)
ax4.axhspan(temperatures.min(), temperatures.max(), color="grey", alpha=0.15, label="Green's function reference range")
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("Ambient temperature [C]")
ax4.set_title("Convection loading temperature history")
ax4.grid(True)
ax4.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "temperature_history.png"), dpi=150)

fig5, ax5 = plt.subplots(figsize=(12, 6))
for T in temperatures:
    sub = df[df[T_col] == T].sort_values(time_col)
    line, = ax5.plot(sub[time_col], sub[stress_col], linewidth=1, alpha=0.5)
    ax5.plot(sub[time_col], G_eq10(T, sub[time_col].values), "--", color=line.get_color(),
             linewidth=1, label=f"{T:.0f} C (R2={r2_by_T[T]:.4f})")
ax5.set_xlabel("Lag time [s]")
ax5.set_ylabel("Von Mises Stress [MPa]")
ax5.set_title("Green's functions: FEA data (solid) vs Eq.10 fit (dashed)")
ax5.grid(True)
ax5.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "greens_function_fits_all.png"), dpi=150)

print("\nDone.")
