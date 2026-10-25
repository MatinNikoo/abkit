# Case study: a failed split check in Cookie Cats

Cookie Cats is a mobile puzzle game. 90,189 new players were randomized into two versions. The first gate (where you wait or pay to keep playing) sits at level 30 (`gate_30`) or level 40 (`gate_40`). The question was whether moving the gate later hurts retention.

The first thing abkit does is check the split. It failed.

| | gate_30 | gate_40 | p |
|---|---|---|---|
| players | 44,700 | 45,489 | 0.0086 |

This page is about what I did next. A failed split check is easy to overreact to and just as easy to ignore. Everything here comes from [`analysis/cookie_cats_srm.py`](../analysis/cookie_cats_srm.py), with the numbers saved in `results/cookie_cats_srm.json`.

## 1. What p = 0.0086 does and doesn't mean

It means that if assignment really were a fair coin flip, a split at least this lopsided would happen about 1 time in 116. That's unusual enough to investigate, and a fair test can still do it. If you run a hundred healthy experiments, about one of them will look like this. Teams that run many tests often only alert below 0.001, and this test would pass that cutoff.

So the honest starting point is a warning sign. What follows is whether the data can say more.

## 2. A natural placebo in the game design

The gate sits at level 30 or 40. A player who played fewer than 29 game rounds can't have finished 29 levels, so they never reached either gate. For those players the two versions were the exact same game. That gives two checks the treatment can't touch:

* The split among those players should be 50/50 if assignment worked.
* Their retention should be the same in both groups (a placebo test).

This rests on one assumption, that a level takes at least one round. That's how the game works, but I can't verify it from the data.

## 3. Where the extra players are

![Where the imbalance is](../figures/cookie_cats_srm.png)

| Players | gate_30 | gate_40 | share in gate_40 | split p |
|---|---|---|---|---|
| never reached a gate (under 29 rounds) | 27,742 | 28,565 | 50.7% | 0.0005 |
| could have reached one (29 or more) | 16,958 | 16,924 | 49.9% | 0.85 |

Most of the imbalance sits in players who never reached a gate (gate_40 has 823 more of them). Among players who could have, the split is almost even (gate_40 has 34 fewer). It isn't perfectly clean, since the 29 to 39 round bucket is off the other way (3,392 vs 3,097, p = 0.00025) and some of the low round buckets are off by more than others. The share of low activity players also differs between the groups (62.1% vs 62.8%, p = 0.023), which shouldn't happen when the treatment can't affect them.

I first suspected players with zero rounds (installed, never played), which would point straight at install logging. They're part of it but not most of it. They split 1,937 vs 2,057, about 15% of the excess. Most of it is players with 1 to 28 rounds.

I picked the 29 round cut from the game design before looking, but I did look at other cuts afterwards, and the pattern holds across them (for example under 20 rounds p = 0.0001, 20 and up p = 0.80). The full curve is in the results file. Splitting by user id deciles shows nothing (p = 0.61), so there's no sign the problem was a few bad days, but user ids may not track install time.

Players with 29 to 39 rounds are split unevenly too (gate_40 has fewer), but that looks like the treatment itself, since gate_30 players stall around level 30 and pile up there. I can't fully separate that from a second problem, which is why only the 29+ total is used as a split check.

## 4. What this points to, and what it can't prove

A uniformly skewed coin (say 50.4% to gate_40 for everyone) would spread the extra players across all activity levels. Instead the extra players are mostly low activity ones, which suggests something that adds or drops low engagement records in one group, such as install attribution, a logging difference or how test users and reinstalls were excluded. That argues against a broken randomizer without ruling one out. The low round buckets don't line up neatly either (1 to 4 rounds p = 0.0046, 5 to 9 p = 0.53, 10 to 19 p = 0.023, 20 to 28 p = 0.84), so I wouldn't read more into the pattern than that.

It can't prove any of that. Chance can still produce this pattern, and the data has no timestamps, devices or logs to check against.

## 5. How much the retention result depends on it

The headline finding is that gate_40 lowered 7 day retention, 19.0% vs 18.2%, a drop of 0.82 points (p = 0.0016). If gate_40 has extra low activity records, part of that gap is just a different mix of players. Here is the 7 day retention result under different readings of the problem.

| Analysis | Change (points) | 95% CI | p |
|---|---|---|---|
| all players, as reported | -0.82 | -1.33 to -0.31 | 0.0016 |
| drop players with 0 rounds | -0.81 | -1.34 to -0.28 | 0.0026 |
| only players with 29+ rounds (the clean, balanced group) | -0.97 | -2.02 to +0.08 | 0.071 |
| reweight the two activity groups to their pooled shares | -0.53 | -0.98 to -0.09 | 0.018 |
| effect only where a gate was possible, on the whole population | -0.36 | -0.76 to +0.03 | 0.071 |
| worst case: the 789 extra gate_40 records were all retained | -2.26 | -2.77 to -1.76 | <0.0001 |
| best case: the 789 extra gate_40 records were all not retained | -0.50 | -1.01 to +0.01 | 0.056 |

The placebo among players who never reached a gate shows 7 day retention was 4.12% vs 3.84% (-0.27 points, p = 0.098). Not significant, but in the direction you'd expect if gate_40 has extra low engagement players mixed in.

A rough split of the -0.82 point gap puts about -0.29 from gate_40 having more low activity players, -0.17 from the small difference inside that group, and -0.36 from the players who could actually reach a gate.

1 day retention doesn't move in any version of the analysis.

## 6. My read

Every analysis puts gate_40 below gate_30 on 7 day retention, so I trust the direction. Among players who could actually hit a gate, retention was about 1 point lower (43.4% vs 42.4%).

The p value rises from 0.0016 to 0.07 in the group where the split is clean. Mostly that is lost data, because that group is only 38% of the players. Its estimated drop is slightly larger than the full sample one (0.97 vs 0.82 points), so the effect didn't shrink. About a third of the full sample gap does come from the different mix of players.

At the company, before shipping either version, I'd pull the assignment and install logs for low activity players in both groups. If extra records turn up in gate_40, I'd go with the clean group answer (-0.97 points among players who reached a gate, -0.36 points spread over everyone). If nothing turns up, the full sample answer stands with a note about the split.

## About the sequential replay

The README also replays this test as if someone had checked it 50 times. The data has no timestamps, so that replay puts players in a random shuffled order. It shows how an always valid test behaves when the order is pure luck. It can't show what would have happened in real time. Over 1,000 shuffles the test called the 7 day result every time, stopping after 22% to 100% of the players (5th to 95th percentile, median 78%).

## Rerun the readout from the command line

After `python analysis/get_data.py`, one command rebuilds the health checks, retention results, guardrail and decision from [examples/cookie_cats.toml](../examples/cookie_cats.toml), and `--html` or `--json` writes them to a file. The readout in `examples/` uses rounds capped at the 99th percentile as its third metric and the config uses raw rounds, so only that row differs.

```bash
abkit report --config examples/cookie_cats.toml
```

[examples/cookie_cats_plan.md](../examples/cookie_cats_plan.md) is what the plan for this test could have looked like. A plan for 7 day retention with a 5% relative effect worth finding needs about 26,800 players per group, and this test had about 45,000, so its size was never the problem.
