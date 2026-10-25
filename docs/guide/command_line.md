# The command line

Four commands cover the usual work. `report` analyzes a finished test, `plan` writes the plan before it starts, `size` answers "how many users", and `aa` checks that a metric behaves. `abkit presets` lists the guardrail defaults.

## Data

A file with one row per user and a column for the group. CSV, TSV, Parquet, Excel (`--sheet NAME` picks a sheet), JSON and JSON lines all work, and `.gz` text files too. Parquet needs `pyarrow` and Excel needs `openpyxl` (`pip install "abkit[files]"`).

`--where "country == 'US'"` keeps the rows a pandas query matches, which is the quick way to drop bots or look at one platform.

Bad input stops with one line that says what to change.

```
$ abkit report data.csv --variant grup ...
error: no variant column called 'grup'. Did you mean 'group'?
$ abkit report data.csv ... --metrics country
error: 'country' isn't numeric (it holds values like 'US', 'DE'). A metric needs numbers or True/False; recode the column first.
```

## report

```bash
abkit report data.csv --variant group --control A --treatment B \
    --metrics orders revenue --primary orders \
    --covariate orders_last_month --segment country \
    --guardrail load_ms:latency refunds:0.1 --lower-is-better refunds \
    --markdown readout.md --html readout.html --json numbers.json --plot effects.png
```

With no output flag it prints to the terminal with health checks first, then every metric, the adjusted p values, guardrails, segments and the decision. With `--markdown`, `--html`, `--json` or `--plot` it writes files instead. A dash as the file name (`--json -`) prints to the terminal, so `abkit report ... --json - | jq .decision` works in a pipeline.

The decision rule is on the [readout page](readout.md). The `--json` file holds every number the report is built from.

Other options

| Option | What it does |
|---|---|
| `--covariate A B` | pre period columns, used for CUPED and checked for balance |
| `--cluster COL` | the column that was randomized when rows aren't independent |
| `--split 0.5` | the share you meant to put in treatment, for the split check |
| `--alpha 0.05`, `--srm-threshold 0.01` | the significance level, and when the split check fails |
| `--correction bh\|holm\|none` | how p values are adjusted across metrics and segments |
| `--prior-sd S` | adds the chance the treatment is better, with a prior from past tests |
| `--lower-is-better M ...` | metrics where down is good |
| `--title TEXT` | the title of the written readout |

## Guardrails and presets

`--guardrail METRIC:MARGIN` takes the worst relative change you would accept, so `retention_1:0.01` allows a 1% drop. `METRIC:PRESET` uses a named default, which also sets the direction.

| Preset | Margin | Bad direction |
|---|---|---|
| latency | 5% | up |
| errors | 10% | up |
| crashes | 5% | up |
| unsubscribes | 10% | up |
| retention | 1% | down |
| reliability | 0.5% | down |
| engagement | 2% | down |
| revenue | 2% | down |

These are starting points. The right margin depends on what a change costs you, so look at them and change them. In Python they live in `abkit.GUARDRAILS`.

## A config file for repeated analyses

Every option can sit in a TOML or JSON file. The command line wins over the file, so one file can describe the experiment and a flag can change one thing.

```toml
data = "results.parquet"
variant = "group"
control = "A"
treatment = "B"
metrics = ["orders", "revenue"]
primary = "orders"
covariate = ["orders_last_month"]

[guardrail]
load_ms = "latency"
```

```bash
abkit report --config experiment.toml --html readout.html
abkit report --config experiment.toml --alpha 0.01      # same file, one change
```

Keys are the option names with underscores or dashes. A key that no command knows is an error, so a typo doesn't get ignored. A table named after a command (`[report]`, `[plan]`) applies to that command only, so the same file can feed `report` and `plan`. [examples/cookie_cats.toml](../../examples/cookie_cats.toml) is a working one.

## plan

Writes a plan to read back later, before any result exists. See [planning](planning.md) for what is in it.

```bash
abkit plan --name "Gate at 40" --primary retention_7 --baseline 0.19 --relative-mde 0.05 \
    --guardrail retention_1:retention --secondary sum_gamerounds --sequential group --looks 4 \
    --daily-users 3000 --out plan.md
```

## size

```bash
abkit size --baseline 0.19 --mde 0.01                      # users per group for a conversion rate
abkit size --baseline 0.19 --relative-mde 0.05 --cuped-rho 0.5
abkit size --sd 12.5 --mde 0.5                             # an average
abkit size --pilot pilot.csv --numerator clicks --denominator sessions --mde 0.01   # a ratio metric
abkit size --baseline 0.19 --users 20000                   # what 20,000 per group can find
abkit size --baseline 0.19 --mde 0.01 --daily-users 4000   # and how many days
```

`--json -` prints the answer as JSON.

## aa

```bash
abkit aa data.csv --metric revenue [--cluster user_id] [--sims 1000] [--json -]
```

Exit codes are 0 for success and 2 for bad input, so a script can tell them apart.
