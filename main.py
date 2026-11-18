"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads every <id>.in case file from the input directory and writes a
<id>.out forecast for each one: 23 rows (hours 2..24 of the UTC release day),
five quantiles per row, in the case's `levels` order.

Method (fitted at run time from the case's own `comparables`, so it adapts
to whatever launch history the case ships):

  1. For each comparable launch, take its release-day hourly viewers and
     three pre-release signals measured strictly before its `day_start`:
     the hour-1 print, the mean hourly viewers of the last 48 hours, and
     the peak hourly viewers of the last 24 hours.
  2. For every forecast hour h, fit an ordinary least-squares regression
     of log(viewers at h) on those three signals (in log space).  The
     launch-day curve is strongly scale-compressed: hour 1 spans orders
     of magnitude across launches while the evening peak sits in a much
     narrower band, so the regression's slope on log hour-1 falls from
     ~1.0 in the small hours to ~0.1 by the evening.  A flat
     "same shape as hour 1" prior misses that; the regression does not.
  3. The forecast centre is the regression prediction.  The spread is
     the empirical quantiles of that hour's regression residuals across
     the comparables, so the interval width follows the data instead of
     a guessed random-walk sigma -- but scaled by SPREAD_SCALE (0.45):
     the residuals are heavy-tailed, so the raw tail quantiles of only
     ~26 residuals overstate the error a fresh launch sees, and the
     unscaled 90% band covered 91 of 92 practice hours instead of the
     ~83 an honest band should.
  3b. Evening shrinkage (hours >= BLEND_MIN_HOUR = 15): the regression's
     slope on log hour-1 decays from ~1.0 in the small hours to ~0.1-0.3
     by the evening, so for a launch much bigger than any comparable its
     evening forecast saturates near a fixed level whatever hour 1 says
     (on a synthetic 3x/5x Ragnarok the regression alone puts the
     evening peak at 0.37x/0.23x hour 1, below every launch in the data).
     For those hours the centre is shrunk 25% toward the comparables'
     median launch shape -- hour 1 times the median evening/hour-1
     ratio -- which pulls big launches' evenings up and small launches'
     down toward the typical curve.  Leave-one-case-out: a blend weight
     picked on any three practice cases (0.2-0.3) improves the held-out
     case on three of the four launches (Wolverine, Red Dead 2,
     Ragnarok) and only worsens Spider-Man 2, whose deep trough the
     median shape does not share; the shrinkage is therefore applied
     only from hour 15, leaving the well-tracked early hours (slope
     ~1.0 on hour 1) to the regression alone.
  4. Comparables are restricted to launches whose research note marks
     them `Tier: comparable`.  Early-access launches (their release day
     is really day 4 of play) and weak analogues (free-to-play launches,
     slow burns) are left out, as is the target game's own pre-release
     category history, which is not a launch at all.  If too few
     comparable-tier launches survive, the filter relaxes to every
     launch except the target's own history.

Fallbacks keep every hour valid: fewer predictors if the three-predictor
fit is singular or starved of points, then the comparables' median
log-ratio shape, then the house persistence forecast.  No network, no
clock, no randomness: the same input bytes always produce the same
output bytes.  Python 3.13 standard library only.
"""

import json
import math
import os
import re
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The levels every case in this competition asks for.  Anything else is
# handled by the same code paths (empirical quantiles interpolate, and the
# persistence fallback inverts the normal CDF), so nothing here assumes
# this list.
STANDARD_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the standard levels, for the last-resort
# persistence fallback only.
STANDARD_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
PERSIST_SIGMA = 0.15

MIN_POINTS_FULL = 12      # comparables needed for the 3-predictor fit
MIN_POINTS_TWO = 10       # comparables needed for the 2-predictor fit
MIN_POINTS_ONE = 8        # comparables needed for the 1-predictor fit
MIN_POINTS_SHAPE = 8      # comparables needed for the median-shape fallback
MIN_DAY_HOURS = 20        # release-day hours a comparable must report
MAX_VIEWERS = 1_000_000_000
# Calibration for the empirical-quantile spread (see forecast_hour): the
# raw residual quantiles overstate a new launch's error, so the deviations
# from the centre are scaled by this factor before they become quantiles.
# 0.45 (was 0.55): after the evening shrinkage below, the centre sits
# closer to the truth on the practice launches, so the honest 90% band
# needs less width; at 0.45 the band still covers 83 of 92 practice
# hours, the honest ~90%.
SPREAD_SCALE = 0.45
# Evening shrinkage (see docstring 3b): how far the regression centre for
# hours >= BLEND_MIN_HOUR moves toward the median launch shape.
BLEND_WEIGHT = 0.25
BLEND_MIN_HOUR = 15


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def parse_time(text):
    """ISO-8601 timestamp (with a Z suffix) to an aware datetime."""
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_of(value):
    return math.log(max(1.0, float(value)))


def mean_of(values):
    return sum(values) / len(values)


def sample_quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already sorted sample."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = tau * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def normal_quantile(tau):
    """Inverse standard-normal CDF by bisection on math.erf (deterministic)."""
    if tau <= 0.0:
        return -8.0
    if tau >= 1.0:
        return 8.0
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def least_squares(design, target):
    """OLS fit of target on design rows (no intercept column supplied).

    Returns (coefficients_with_intercept, residuals) or None.
    """
    k = len(design[0]) + 1
    normal = [[0.0] * k for _ in range(k)]
    moment = [0.0] * k
    for row, y in zip(design, target):
        full = [1.0] + list(row)
        for i in range(k):
            moment[i] += full[i] * y
            for j in range(k):
                normal[i][j] += full[i] * full[j]
    beta = solve_linear(normal, moment)
    if beta is None:
        return None
    residuals = []
    for row, y in zip(design, target):
        fitted = beta[0] + sum(b * x for b, x in zip(beta[1:], row))
        residuals.append(y - fitted)
    return beta, residuals


def round_half_up(value):
    """Round to 2 decimals, halves away from zero; whole numbers as ints."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100.0
    return int(out) if float(out).is_integer() else out


# --------------------------------------------------------------------------
# case data: hours, pre-release signals, comparable launches
# --------------------------------------------------------------------------

def hourly_points(case):
    """Hourly (timestamp, viewers, channels) rows before the release day.

    Prefers history_hourly; falls back to bucketing the per-minute
    history into whole hours.
    """
    rows = case.get("history_hourly") or []
    parsed = []
    for row in rows:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        parsed.append((parse_time(row["from"]), float(viewers)))
    if parsed:
        return parsed
    buckets = {}
    for row in case.get("history") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        stamp = parse_time(row["at"])
        key = stamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(float(viewers))
    return [(key, mean_of(vals)) for key, vals in sorted(buckets.items())]


def pre_release_signals(points, day_start):
    """Log-space hype signals from the hours strictly before day_start."""
    last_48 = [v for t, v in points if -48 * 3600 <= (t - day_start).total_seconds() < 0]
    last_24 = [v for t, v in points if -24 * 3600 <= (t - day_start).total_seconds() < 0]
    return {
        "log_pre48": log_of(mean_of(last_48)) if last_48 else None,
        "log_pre24max": log_of(max(last_24)) if last_24 else None,
    }


def release_day_hours(comp):
    """{hour: viewers} for the comparable's UTC release day, hours 1..24."""
    day_start = parse_time(comp["day_start"])
    hours = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        offset = (parse_time(row["from"]) - day_start).total_seconds() / 3600.0
        hour = int(offset) + 1
        if 1 <= hour <= 24:
            hours[hour] = float(viewers)
    return day_start, hours


def research_tier(comp):
    """The 'Tier: ...' tag in the comparable's research note, if any."""
    note = comp.get("unlock") or ""
    match = re.search(r"Tier:\s*([a-z\-]+)", note)
    return match.group(1) if match else "unknown"


def collect_comparables(case):
    """Usable comparable launches, best tier first.

    Each entry carries its release-day hours plus the three pre-release
    signals (log hour-1, log 48h mean, log 24h peak).  The target game's
    own pre-release category history is never a launch and is excluded.
    """
    usable = []
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or "").lower()
        if "grand theft auto vi" in name and "category" in name:
            continue
        if research_tier(comp) == "target-history":
            continue
        try:
            day_start, hours = release_day_hours(comp)
        except (KeyError, TypeError, ValueError):
            continue
        if 1 not in hours or len(hours) < MIN_DAY_HOURS:
            continue
        points = []
        for row in comp.get("hourly") or []:
            viewers = row.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            points.append((parse_time(row["from"]), float(viewers)))
        signals = pre_release_signals(points, day_start)
        if signals["log_pre48"] is None or signals["log_pre24max"] is None:
            continue
        usable.append({
            "name": comp.get("name") or "",
            "tier": research_tier(comp),
            "hours": hours,
            "log_h1": log_of(hours[1]),
            "log_pre48": signals["log_pre48"],
            "log_pre24max": signals["log_pre24max"],
        })
    usable.sort(key=lambda entry: entry["name"])
    preferred = [entry for entry in usable if entry["tier"] == "comparable"]
    if len(preferred) >= MIN_POINTS_FULL:
        return preferred
    return usable


# --------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------

def forecast_hour(hour, comparables, levels, target, p1):
    """Five quantiles for one forecast hour, best available method first."""
    have = [c for c in comparables if hour in c["hours"]]
    predictors = (
        (("log_h1", "log_pre48", "log_pre24max"), MIN_POINTS_FULL),
        (("log_h1", "log_pre48"), MIN_POINTS_TWO),
        (("log_h1",), MIN_POINTS_ONE),
    )
    for keys, minimum in predictors:
        if len(have) < minimum:
            continue
        design = [[c[key] for key in keys] for c in have]
        target_log = [log_of(c["hours"][hour]) for c in have]
        fitted = least_squares(design, target_log)
        if fitted is None:
            continue
        beta, residuals = fitted
        centre = beta[0] + sum(b * target[key] for b, key in zip(beta[1:], keys))
        if hour >= BLEND_MIN_HOUR and len(have) >= MIN_POINTS_SHAPE:
            # Evening shrinkage: pull the centre toward the median launch
            # shape (hour 1 times the median evening/hour-1 ratio).  The
            # regression's slope on log hour-1 has decayed to ~0.1-0.3 by
            # the evening, so its evening forecast saturates for launches
            # bigger than any comparable; the median shape keeps the
            # evening tied to the launch's own scale.
            ratios = sorted(log_of(c["hours"][hour]) - c["log_h1"] for c in have)
            shape = log_of(p1) + sample_quantile(ratios, 0.5)
            centre = (1.0 - BLEND_WEIGHT) * centre + BLEND_WEIGHT * shape
        ordered = sorted(residuals)
        # The residuals are heavy-tailed: a few launches deviate far more
        # than a typical new launch does, so the raw tail quantiles of
        # only ~26 residuals overstate the error a fresh launch will see.
        # On the practice launches the unscaled 5%-95% band covered 91 of
        # 92 hours (it should cover about 90%); scaling the deviations by
        # SPREAD_SCALE brings that to 83 of 92, i.e. an honest 90% band,
        # and the pinball loss falls on every practice launch.  The factor
        # is one global calibration, not tuned per hour or per launch.
        return [centre + SPREAD_SCALE * sample_quantile(ordered, tau)
                for tau in levels]
    if len(have) >= MIN_POINTS_SHAPE:
        ratios = sorted(log_of(c["hours"][hour]) - c["log_h1"] for c in have)
        centre = log_of(p1)
        return [centre + SPREAD_SCALE * sample_quantile(ratios, tau)
                for tau in levels]
    zetas = [STANDARD_Z[STANDARD_LEVELS.index(tau)] if tau in STANDARD_LEVELS
             else normal_quantile(tau) for tau in levels]
    centre = log_of(p1)
    return [centre + z * PERSIST_SIGMA * math.sqrt(hour - 1) for z in zetas]


def clean_quantiles(log_values):
    """Log-space quantiles -> clamped, rounded, non-decreasing viewer counts."""
    counts = []
    for value in log_values:
        number = math.exp(value)
        if not math.isfinite(number):
            number = MAX_VIEWERS
        counts.append(round_half_up(min(max(number, 1.0), float(MAX_VIEWERS))))
    for i in range(1, len(counts)):
        if counts[i] < counts[i - 1]:
            counts[i] = counts[i - 1]
    return counts


def forecast_case(case):
    levels = list(case.get("levels") or STANDARD_LEVELS)
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    day_start = parse_time(case["day_start"])
    signals = pre_release_signals(hourly_points(case), day_start)
    target = {
        "log_h1": log_of(p1),
        "log_pre48": signals["log_pre48"],
        "log_pre24max": signals["log_pre24max"],
    }
    if target["log_pre48"] is None or target["log_pre24max"] is None:
        # No pre-release history at all: the hype signals are unknown,
        # so fall back to the hour-1 print as the only scale signal.
        target["log_pre48"] = target["log_h1"]
        target["log_pre24max"] = target["log_h1"]
    comparables = collect_comparables(case)
    wanted = [row["hour"] for row in case.get("forecast") or []]
    if not wanted:
        wanted = list(range(2, 25))
    rows = []
    for hour in wanted:
        log_q = forecast_hour(hour, comparables, levels, target, p1)
        rows.append({"hour": hour, "q": clean_quantiles(log_q)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
