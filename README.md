[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/doog87654321-eng/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/doog87654321-eng/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01 — forecast GTA VI's launch-day Twitch viewers

A single-file Python 3.13 program (`main.py`, standard library only, no
network) that forecasts the Grand Theft Auto VI Twitch category's hourly
average viewers for hours 2–24 of launch day, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour. It reads every `<id>.in` case
file from the input directory and writes a `<id>.out` forecast for each.

## How it works

The program fits its model at run time from the case's own `comparables`
(38 past AAA launches in the practice cases, 42 in the final), so it
adapts to whatever launch history the case ships:

1. **Per-hour log regression.** For each forecast hour, it regresses
   log(viewers) across comparable launches on three log-space signals
   measured strictly before each launch's `day_start`: the hour-1 print,
   the mean hourly viewers of the last 48 hours, and the peak hourly
   viewers of the last 24 hours. The key fact the regression captures is
   scale compression: hour 1 spans orders of magnitude across launches
   (164 for Silksong, 875k for Elden Ring) while the evening peak sits in
   a much narrower band, so the fitted slope on log hour-1 falls from
   ~1.0 in the small hours to ~0.1 by the evening. A flat "same shape as
   hour 1" prior misses that; the regression does not.
2. **Empirical residual spread.** The interval width for each hour is the
   empirical quantiles of that hour's regression residuals across the
   comparables, so the spread follows the data instead of a guessed
   random-walk sigma.
3. **Tier filter.** Only launches whose research note marks them
   `Tier: comparable` are used. Early-access launches (their release day
   is really day 4 of play), weak analogues (free-to-play launches, slow
   burns) and the target game's own pre-release category history (not a
   launch at all) are excluded; if too few comparable-tier launches
   survive, the filter relaxes to every launch except the target's own
   history.
4. **Evening shrinkage.** From hour 15 the regression's slope on log
   hour-1 has decayed to ~0.1–0.3, so for a launch much bigger than any
   comparable its evening forecast saturates near a fixed level whatever
   hour 1 says. Those hours' centres are therefore shrunk 25% toward
   the comparables' median launch shape (hour 1 × the median
   evening/hour-1 ratio), which pulls big launches' evenings up and
   small launches' down toward the typical curve. Leave-one-case-out:
   a weight picked on any three practice launches (0.2–0.3) improves
   the held-out launch on three of the four, and only worsens
   Spider-Man 2 (+0.007), whose deep trough the median shape does not
   share — so the shrinkage applies only from hour 15, leaving the
   well-tracked early hours to the regression alone. The spread is
   scaled by 0.45 (after the shrinkage the centre sits closer to the
   truth, so the honest 90% band needs less width; it still covers
   83 of 92 practice hours).
5. **Fallbacks keep every hour valid:** fewer predictors if a fit is
   singular or starved of points, then the comparables' median log-ratio
   shape, then the house persistence forecast. No clock, no randomness:
   the same input bytes always produce the same output bytes.

The 48h/24h signal windows are deliberate: the final case's own history
only spans about two days (the house started recording the GTA VI
category on 2026-11-17), so a 7-day hype signal — which scores slightly
better on the practice cases — would be silently truncated in the final
and over-forecast. Verified: truncating the practice histories to 49
hours, or dropping `history_hourly` entirely (per-minute history is
bucketed into hours instead), leaves the scores unchanged.

## Practice results

Practice answers were recovered from SullyGnome's public chart API
("Average viewers", same source the kit's NOTICE points to) by anchoring
on each case's known hour-1 print. Mean pinball loss on the log scale
(lower is better; house baseline ≈ 0.387):

| case | game | model | house baseline |
| --- | --- | --- | --- |
| p00 | Marvel's Wolverine | 0.128 | 0.744 |
| p01 | Red Dead Redemption 2 | 0.126 | 0.287 |
| p02 | God of War Ragnarök | 0.108 | 0.215 |
| p03 | Marvel's Spider-Man 2 | 0.201 | 0.303 |
| **mean** | | **0.141** | **0.387** |

Typical median miss 18–37% (baseline 42–189%); 90% interval coverage
0.78–0.96 (overall 83 of 92 hours). Runtime ≈ 0.07 s per case, far inside the 8 s limit.

## Usage

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with `/input` and `/output`.
Output is one JSON object per case, `{"hours": [{"hour": 2, "q": [...]},
...]}`, exactly 23 rows, hours 2–24 each once, five finite quantiles per
row, non-decreasing, each between 1 and 1,000,000,000, under 65,536 bytes.
