"""Verlet crossover under deterministic and explicitly stochastic rounding."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"
T = 100.0
K_VALUES = range(6, 19)  # N = 64 ... 262144 steps
STOCHASTIC_STEP_COUNTS = sorted({round(2**x) for x in np.arange(6, 17.01, 0.5)})
REPLICATES = 128
PRECISION_BITS = (8, 10, 12, 16, 20, 24)
DURATION_VALUES = (25.0, 50.0, 100.0, 200.0, 400.0)
DURATION_BITS = 16
DURATION_REPLICATES = 512
DURATION_H_DENOMINATORS = tuple(range(35, 201, 5))
PRECISION_BOOTSTRAPS = 2000
DURATION_BOOTSTRAPS = 2000


def initial_ensemble():
    theta = np.random.default_rng(2026).uniform(0.0, 2 * np.pi, REPLICATES)
    return np.cos(theta), np.sin(theta)


def exact_state(q0, p0):
    return exact_state_at(q0, p0, T)


def exact_state_at(q0, p0, duration):
    return (
        np.cos(duration) * q0 + np.sin(duration) * p0,
        -np.sin(duration) * q0 + np.cos(duration) * p0,
    )


def verlet_matrix(n, duration):
    h = duration / n
    return np.array(
        [[1.0 - h * h / 2.0, h], [-h * (1.0 - h * h / 4.0), 1.0 - h * h / 2.0]]
    )


def deterministic_bias_vector(n, duration):
    """Matrix bias for the idealized model with exact stage arithmetic."""
    m_n = np.linalg.matrix_power(verlet_matrix(n, duration), n)
    rotation = np.array(
        [[np.cos(duration), np.sin(duration)], [-np.sin(duration), np.cos(duration)]]
    )
    delta = m_n - rotation
    return delta @ np.array([1.0, 0.0])


def deterministic_truncation_rms(n, duration):
    return float(np.linalg.norm(deterministic_bias_vector(n, duration)))


def state_error_decomposition(q, p, qe, pe):
    errors = np.column_stack((q.astype(np.float64) - qe, p.astype(np.float64) - pe))
    mean_error = errors.mean(axis=0)
    empirical_bias2 = float(mean_error @ mean_error)
    empirical_variance = float(np.mean(np.sum((errors - mean_error) ** 2, axis=1)))
    mse = empirical_bias2 + empirical_variance
    return errors, empirical_bias2, empirical_variance, mse


def state_errors(q, p, q0, p0):
    qe, pe = exact_state(q0, p0)
    return np.sqrt((q.astype(np.float64) - qe) ** 2 + (p.astype(np.float64) - pe) ** 2)


def relative_energy_errors(q, p):
    return np.abs(q.astype(np.float64) ** 2 + p.astype(np.float64) ** 2 - 1.0)


def integrate_precision(q0, p0, n, dtype):
    h = T / n
    q, p = q0.astype(dtype), p0.astype(dtype)
    half_h, full_h = dtype(0.5 * h), dtype(h)
    for _ in range(n):
        p_half = (p - half_h * q).astype(dtype)
        q_next = (q + full_h * p_half).astype(dtype)
        p_next = (p_half - half_h * q_next).astype(dtype)
        q, p = q_next, p_next
    return q, p


def stochastic_round(x, bits, rng):
    """Unbiased stochastic rounding to adjacent p-bit binary floats.

    For exact x between representable neighbors lo and hi, returns hi with
    probability (x-lo)/(hi-lo), and lo otherwise. The exponent is unbounded.
    this isolates significand-rounding effects from exponent-range effects.
    """
    x = np.asarray(x, dtype=np.float64)
    mantissa, exponent = np.frexp(x)
    scaled = np.ldexp(mantissa, bits)
    lower_units = np.floor(scaled)
    probability_up = scaled - lower_units
    units = lower_units + (rng.random(x.shape) < probability_up)
    return np.ldexp(units, exponent - bits)


def integrate_stochastic_rounding(n, bits, rng, duration=T, replicates=REPLICATES):
    """Stochastically round each kick/drift state update in the Verlet map."""
    h = duration / n
    # Exactly representable initial condition avoids an h-independent input error.
    q, p = np.ones(replicates), np.zeros(replicates)
    # Retain coefficients in FP64 to isolate stochastic rounding of the state.
    h_store = h
    half_h_store = 0.5 * h
    for _ in range(n):
        p_half = stochastic_round(p - half_h_store * q, bits, rng)
        q_next = stochastic_round(q + h_store * p_half, bits, rng)
        p_next = stochastic_round(p_half - half_h_store * q_next, bits, rng)
        q, p = q_next, p_next
    return q, p


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def duration_seed_map():
    """Keep unique legacy streams and deterministically re-seed collisions."""
    keys = [
        (duration, denominator)
        for duration in DURATION_VALUES
        for denominator in DURATION_H_DENOMINATORS
    ]
    legacy = {key: 73019 + int(key[0]) * 100 + int(key[0]) * key[1] for key in keys}
    reserved = set(legacy.values())
    assigned = set()
    seeds = {}
    reassigned = 0
    for duration, denominator in keys:
        seed = legacy[(duration, denominator)]
        if seed in assigned:
            reassigned += 1
            salt = 0
            while True:
                candidate = int(
                    np.random.SeedSequence(
                        [73019, int(duration), denominator, salt]
                    ).generate_state(1, dtype=np.uint64)[0]
                )
                if candidate not in assigned and candidate not in reserved:
                    seed = candidate
                    break
                salt += 1
        seeds[(duration, denominator)] = seed
        assigned.add(seed)
    return seeds, reassigned


def main():
    OUT.mkdir(exist_ok=True)
    q0, p0 = initial_ensemble()
    precision_rows, stochastic_rows = [], []
    stochastic_errors = {bits: {} for bits in PRECISION_BITS}

    # Direct state-rounding runs distinguish a real FP32 crossover from the
    # truncation-dominated FP64 control over the same duration and step grid.
    for label, dtype in (("fp32", np.float32), ("fp64", np.float64)):
        for k in K_VALUES:
            n = 2**k
            q, p = integrate_precision(q0, p0, n, dtype)
            err = state_errors(q, p, q0, p0)
            energy = relative_energy_errors(q, p)
            precision_rows.append(
                {
                    "precision": label,
                    "steps": n,
                    "h": T / n,
                    "replicates": REPLICATES,
                    "rms_state_error": float(np.sqrt(np.mean(err**2))),
                    "median_state_error": float(np.median(err)),
                    "q90_state_error": float(np.quantile(err, 0.9)),
                    "rms_relative_energy_error": float(np.sqrt(np.mean(energy**2))),
                }
            )

    # True stochastic rounding at the three state-update outputs per step.
    for bits in PRECISION_BITS:
        for n in STOCHASTIC_STEP_COUNTS:
            rng = np.random.default_rng(42017 + 1000 * bits + n)
            q, p = integrate_stochastic_rounding(n, bits, rng)
            qe, pe = exact_state(np.ones(REPLICATES), np.zeros(REPLICATES))
            errors, sample_bias2, sample_var, mse = state_error_decomposition(
                q, p, qe, pe
            )
            stochastic_errors[bits][n] = errors
            energy = relative_energy_errors(q, p)
            exact_bias2 = float(
                deterministic_bias_vector(n, T) @ deterministic_bias_vector(n, T)
            )
            stochastic_rows.append(
                {
                    "significand_bits": bits,
                    "unit_roundoff": 2.0**-bits,
                    "steps": n,
                    "h": T / n,
                    "replicates": REPLICATES,
                    "rms_state_error": float(np.sqrt(mse)),
                    "empirical_mse": mse,
                    "exact_bias_squared": exact_bias2,
                    "sample_bias_squared": sample_bias2,
                    "empirical_variance_component": sample_var,
                    "variance_trace_unbiased": float(
                        np.var(errors, axis=0, ddof=1).sum()
                    ),
                    "estimated_expected_mse": exact_bias2
                    + float(np.var(errors, axis=0, ddof=1).sum()),
                    "rms_relative_energy_error": float(np.sqrt(np.mean(energy**2))),
                }
            )

    write_csv(OUT / "precision_sweep.csv", precision_rows)
    write_csv(OUT / "stochastic_sweep.csv", stochastic_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for precision in ("fp32", "fp64"):
        g = [r for r in precision_rows if r["precision"] == precision]
        ax.loglog(
            [r["h"] for r in g],
            [r["rms_state_error"] for r in g],
            "o-",
            label=precision,
        )
        best = min(g, key=lambda r: r["rms_state_error"])
        ax.scatter([best["h"]], [best["rms_state_error"]], s=65, marker="*", zorder=4)
    ax.set(
        xlabel="step size h",
        ylabel="RMS final-state error",
        title="Direct floating-point state rounding",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "precision_crossover.png", dpi=170)

    optima = []
    for bits in PRECISION_BITS:
        g = [r for r in stochastic_rows if r["significand_bits"] == bits]
        best = min(g, key=lambda r: r["estimated_expected_mse"])
        optima.append(
            (2.0**-bits, best["h"], np.sqrt(best["estimated_expected_mse"]), bits)
        )

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for bits in PRECISION_BITS:
        g = [r for r in stochastic_rows if r["significand_bits"] == bits]
        ax.loglog(
            [r["h"] for r in g],
            np.sqrt([r["estimated_expected_mse"] for r in g]),
            "o-",
            ms=3,
            label=f"s={bits}",
        )
        best = min(g, key=lambda r: r["estimated_expected_mse"])
        ax.scatter(
            [best["h"]],
            [np.sqrt(best["estimated_expected_mse"])],
            s=55,
            marker="*",
            zorder=4,
        )

    fit_optima = [item for item in optima if item[3] >= 10]
    fit_u = np.array([x for x, _, _, _ in fit_optima])
    fit_h = np.array([y for _, y, _, _ in fit_optima])
    slope, intercept = np.polyfit(np.log(fit_u), np.log(fit_h), 1)
    precision_boot_rng = np.random.default_rng(918377)
    precision_boot_h = np.empty((PRECISION_BOOTSTRAPS, len(PRECISION_BITS)))
    for pi, bits in enumerate(PRECISION_BITS):
        group = sorted(
            (r for r in stochastic_rows if r["significand_bits"] == bits),
            key=lambda r: r["steps"],
        )
        h_array = np.array([r["h"] for r in group])
        bias2 = np.array([r["exact_bias_squared"] for r in group])
        errors = np.stack([stochastic_errors[bits][r["steps"]] for r in group])
        indices = precision_boot_rng.integers(
            0, REPLICATES, size=(PRECISION_BOOTSTRAPS, len(group), REPLICATES)
        )
        for bi, idx_by_h in enumerate(indices):
            resampled = np.take_along_axis(errors, idx_by_h[:, :, None], axis=1)
            variance = np.var(resampled, axis=1, ddof=1).sum(axis=1)
            precision_boot_h[bi, pi] = h_array[np.argmin(bias2 + variance)]
    fit_columns = [i for i, bits in enumerate(PRECISION_BITS) if bits >= 10]
    precision_boot_slopes = np.polyfit(
        np.log(fit_u), np.log(precision_boot_h[:, fit_columns]).T, 1
    )[0]
    precision_slope_ci = np.quantile(precision_boot_slopes, [0.025, 0.975]).tolist()
    precision_boot_optima = []
    for pi, bits in enumerate(PRECISION_BITS):
        interval = np.quantile(precision_boot_h[:, pi], [0.025, 0.975]).tolist()
        precision_boot_optima.append(
            {"significand_bits": bits, "h_ci95": [float(v) for v in interval]}
        )
    u_grid = np.geomspace(fit_u.min(), fit_u.max(), 100)
    ax.set(
        xlabel="step size h",
        ylabel="RMS final-state error",
        title="Per-stage stochastic rounding",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stochastic_error_curves.png", dpi=170)
    plt.close(fig)

    # Fit B to stochastic trajectory variance, separate from exact squared bias.
    A_phase = T / 24.0  # Long-duration phase proxy. Exact bias uses the matrix.
    # Hold out s=16 completely: it is the precision used by the duration test.
    fit_rows = [
        r
        for r in stochastic_rows
        if r["significand_bits"] >= 10
        and r["significand_bits"] != DURATION_BITS
        and r["h"] <= 0.2
    ]
    x = np.array([r["unit_roundoff"] ** 2 * T / r["h"] for r in fit_rows])
    y = np.array([r["variance_trace_unbiased"] for r in fit_rows])
    B2 = max(0.0, float(np.dot(x, y) / np.dot(x, x)))
    B = float(np.sqrt(B2))
    predicted = [(u, (B2 * u * u * T / (4 * A_phase * A_phase)) ** 0.2) for u in fit_u]
    # Fixed-B out-of-sample prediction: evaluate the s=16 precision sweep
    # without refitting B or using any duration-sweep observations.
    fixed_b_prediction = []
    for u, h_observed, _, bits in optima:
        if bits != DURATION_BITS:
            continue
        h_pred = (B2 * u * u * T / (4 * A_phase * A_phase)) ** 0.2
        fixed_b_prediction.append(
            {
                "significand_bits": bits,
                "unit_roundoff": u,
                "predicted_h": h_pred,
                "observed_h": h_observed,
                "relative_h_error": abs(h_pred - h_observed) / h_observed,
            }
        )
    heldout = fixed_b_prediction[0]
    fig, fit_ax = plt.subplots(figsize=(7, 4.8))
    for u, h_opt, _, bits in fit_optima:
        fit_ax.scatter([u], [h_opt], s=38, color="black")
        fit_ax.annotate(
            f"s={bits}",
            (u, h_opt),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=9,
        )
    fit_ax.loglog(
        u_grid,
        np.exp(intercept) * u_grid**slope,
        "-",
        label=f"grid-minimum fit: slope {slope:.3f}",
    )
    fit_ax.loglog(
        u_grid, np.exp(intercept) * u_grid**0.4, "--", label="reference slope 0.4"
    )
    fit_ax.scatter(
        [heldout["unit_roundoff"]],
        [heldout["predicted_h"]],
        marker="s",
        s=60,
        facecolors="none",
        edgecolors="black",
        zorder=5,
        label="fixed-B prediction at s=16",
    )
    fit_ax.set(
        xlabel=r"unit roundoff $u=2^{-s}$",
        ylabel="grid-minimizing step h",
        title="Precision scaling of the optimal step",
    )
    fit_ax.grid(True, which="both", alpha=0.25)
    fit_ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "precision_scaling.png", dpi=170)
    plt.close(fig)

    # Independent duration test on the same widened absolute h grid for every T.
    duration_rows = []
    duration_errors = {}
    duration_seeds, duration_seed_collisions_reassigned = duration_seed_map()
    for duration in DURATION_VALUES:
        duration_errors[duration] = {}
        for denominator in DURATION_H_DENOMINATORS:
            h = 1.0 / denominator
            n = round(duration * denominator)
            rng = np.random.default_rng(duration_seeds[(duration, denominator)])
            q, p = integrate_stochastic_rounding(
                n, DURATION_BITS, rng, duration, DURATION_REPLICATES
            )
            qe, pe = exact_state_at(
                np.ones(DURATION_REPLICATES), np.zeros(DURATION_REPLICATES), duration
            )
            errors, sample_bias2, sample_var, mse = state_error_decomposition(
                q, p, qe, pe
            )
            exact_bias = deterministic_bias_vector(n, duration)
            variance_unbiased = float(np.var(errors, axis=0, ddof=1).sum())
            risk = float(exact_bias @ exact_bias) + variance_unbiased
            duration_errors[duration][denominator] = errors
            duration_rows.append(
                {
                    "duration": duration,
                    "significand_bits": DURATION_BITS,
                    "unit_roundoff": 2.0**-DURATION_BITS,
                    "steps": n,
                    "h": h,
                    "replicates": DURATION_REPLICATES,
                    "rms_state_error": float(np.sqrt(mse)),
                    "empirical_mse": mse,
                    "exact_bias_squared": float(exact_bias @ exact_bias),
                    "sample_bias_squared": sample_bias2,
                    "empirical_variance_component": sample_var,
                    "variance_trace_unbiased": variance_unbiased,
                    "estimated_expected_mse": risk,
                }
            )
    write_csv(OUT / "duration_sweep.csv", duration_rows)
    duration_optima = []
    for duration in DURATION_VALUES:
        group = [r for r in duration_rows if r["duration"] == duration]
        best = min(group, key=lambda r: r["estimated_expected_mse"])
        duration_optima.append(
            {
                "duration": duration,
                "h": best["h"],
                "risk_mse": best["estimated_expected_mse"],
                "steps": best["steps"],
            }
        )

    # Bootstrap trajectories to quantify minimum selection and slope uncertainty.
    boot_rng = np.random.default_rng(551902)
    bootstrap_h = np.empty((DURATION_BOOTSTRAPS, len(DURATION_VALUES)))
    for ti, duration in enumerate(DURATION_VALUES):
        denominator_array = np.array(DURATION_H_DENOMINATORS)
        e = np.stack([duration_errors[duration][d] for d in DURATION_H_DENOMINATORS])
        bias2 = np.array(
            [
                r["exact_bias_squared"]
                for r in duration_rows
                if r["duration"] == duration
            ]
        )
        # The trajectories at different step sizes are independent samples.
        # Resample within each grid point, preserving q/p pairing per trajectory.
        indices = boot_rng.integers(
            0,
            DURATION_REPLICATES,
            size=(DURATION_BOOTSTRAPS, len(denominator_array), DURATION_REPLICATES),
        )
        for bi, idx_by_h in enumerate(indices):
            resampled = np.take_along_axis(e, idx_by_h[:, :, None], axis=1)
            var = np.var(resampled, axis=1, ddof=1).sum(axis=1)
            chosen = int(np.argmin(bias2 + var))
            bootstrap_h[bi, ti] = 1.0 / denominator_array[chosen]
    bootstrap_slopes = np.polyfit(
        np.log(np.array(DURATION_VALUES)), np.log(bootstrap_h.T), 1
    )[0]
    duration_slope, duration_intercept = np.polyfit(
        np.log([r["duration"] for r in duration_optima]),
        np.log([r["h"] for r in duration_optima]),
        1,
    )
    slope_ci = np.quantile(bootstrap_slopes, [0.025, 0.975]).tolist()
    for ti, row in enumerate(duration_optima):
        row["h_ci95"] = [
            float(x) for x in np.quantile(bootstrap_h[:, ti], [0.025, 0.975])
        ]
    # Fixed-B duration predictions use B calibrated from T=100, s=10,12,20,24.
    # No s=16 duration observations enter this curve or the prediction table.
    duration_predictions = []
    for row in duration_optima:
        duration = float(row["duration"])
        h_pred = (144.0 * B2 * (2.0 ** (-DURATION_BITS)) ** 2 / duration) ** 0.2
        duration_predictions.append(
            {
                "duration": duration,
                "significand_bits": DURATION_BITS,
                "predicted_h": h_pred,
                "observed_h": float(row["h"]),
                "observed_h_ci95_low": float(row["h_ci95"][0]),
                "observed_h_ci95_high": float(row["h_ci95"][1]),
                "relative_error": float((row["h"] - h_pred) / h_pred),
            }
        )
    write_csv(OUT / "fixed_b_duration_predictions.csv", duration_predictions)
    duration_fit = {
        "significand_bits": DURATION_BITS,
        "predicted_loglog_slope": -0.2,
        "observed_loglog_slope": float(duration_slope),
        "slope_ci95": [float(x) for x in slope_ci],
        "bootstrap_replicates": DURATION_BOOTSTRAPS,
        "common_h_grid": [1.0 / d for d in DURATION_H_DENOMINATORS],
        "B_fit_significand_bits": [10, 12, 20, 24],
        "roundoff_coefficient_B": B,
        "fixed_B_predictions": duration_predictions,
        "bootstrap_resampling": (
            "independent by duration and h, with paired state components "
            "within each trajectory"
        ),
        "optima": duration_optima,
        "rng_seed_scheme": (
            "legacy formula retained for unique conditions; duplicate seeds "
            "are reassigned using SeedSequence([73019, duration, "
            "timestep_denominator, salt])"
        ),
        "rng_seed_collisions_reassigned": duration_seed_collisions_reassigned,
    }
    (OUT / "duration_scaling.json").write_text(
        json.dumps(duration_fit, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ds = np.array([r["duration"] for r in duration_optima])
    hs = np.array([r["h"] for r in duration_optima])
    h_ci = np.array([r["h_ci95"] for r in duration_optima])
    ax.errorbar(
        ds,
        hs,
        yerr=np.maximum(0, np.vstack((hs - h_ci[:, 0], h_ci[:, 1] - hs))),
        fmt="o",
        capsize=3,
        label="grid minima, bootstrap 95% CI",
    )
    grid = np.geomspace(ds.min(), ds.max(), 100)
    ax.loglog(
        grid,
        np.exp(duration_intercept) * grid**duration_slope,
        "-",
        label=f"fit {duration_slope:.3f} [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]",
    )
    h_fixed_b = (144.0 * B2 * (2.0 ** (-DURATION_BITS)) ** 2 / grid) ** 0.2
    ax.loglog(grid, h_fixed_b, "--", label="fixed-B prediction (s=16)")
    ax.set(
        xlabel="duration T",
        ylabel="grid-minimizing step h",
        title=f"Duration scaling, s={DURATION_BITS}",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "duration_scaling.png", dpi=170)

    with (OUT / "crossover_fit.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "phase_coefficient_A": A_phase,
                "roundoff_coefficient_B": B,
                "B_fit_significand_bits": [10, 12, 20, 24],
                "duration_precision_held_out": DURATION_BITS,
                "fixed_B_out_of_sample_predictions": fixed_b_prediction,
                "observed_loglog_slope": float(slope),
                "fit_significand_bits_min": 10,
                "precision_slope_ci95": [float(v) for v in precision_slope_ci],
                "precision_bootstrap_replicates": PRECISION_BOOTSTRAPS,
                "precision_bootstrap_optima": precision_boot_optima,
                "precision_bootstrap_resampling": (
                    "independent by significand precision and timestep, with "
                    "paired state components within each trajectory"
                ),
                "precision_rng_seed_scheme": (
                    "42017 + 1000*significand_bits + step_count "
                    "(collision-free on tested grid)"
                ),
                "predicted_exponent": 0.4,
                "observed_optima": [
                    {"unit_roundoff": u, "h": h, "rms_error": e, "significand_bits": p}
                    for u, h, e, p in optima
                ],
                "model_optima": [{"unit_roundoff": u, "h": h} for u, h in predicted],
                "duration_scaling": duration_fit,
            },
            f,
            indent=2,
        )
    print(
        f"Stochastic-rounding h_opt slope: {slope:.3f} (predicted 0.4), "
        f"phase A={A_phase:.6g}, fitted B={B:.6g}"
    )
    print(
        "Precision h_opt bootstrap 95% CI "
        f"[{precision_slope_ci[0]:.3f}, {precision_slope_ci[1]:.3f}]"
    )
    print(
        f"Duration h_opt slope: {duration_slope:.3f} (predicted -0.2), "
        f"bootstrap 95% CI [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]"
    )


if __name__ == "__main__":
    main()
