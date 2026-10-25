# one command per step; `make all` reruns everything from a clean checkout
PY ?= python3
RUN = OMP_NUM_THREADS=1 nice -n 10 $(PY)

.PHONY: all install test data sims stress new cookie plan golden validation paper clean

all: install data test sims stress new cookie plan golden validation paper

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

data:
	$(PY) analysis/get_data.py

sims:          # the original checks, a few minutes
	$(RUN) simulations/run_simulations.py

stress:        # messy data, about 6 minutes
	$(RUN) simulations/stress.py

new:           # group sequential, CUPAC, power, rank tools, about 30 minutes
	$(RUN) simulations/new_methods.py
	$(RUN) simulations/msprt_rare_events.py

cookie: data
	$(RUN) analysis/cookie_cats.py
	$(RUN) analysis/cookie_cats_srm.py

plan:          # the example plan, same command as in the README
	abkit plan --name "Cookie Cats gate at level 40" --primary retention_7 \
		--hypothesis "Moving the first gate from level 30 to level 40 raises 7 day retention." \
		--baseline 0.19 --relative-mde 0.05 --guardrail retention_1:retention --secondary sum_gamerounds \
		--sequential group --looks 4 --daily-users 3000 --owner Matin --date 2026-10-24 \
		--out examples/cookie_cats_plan.md

golden:
	$(PY) tools/golden_vectors.py

validation:    # the pass/fail table in docs/validation.md, from the saved results
	$(PY) tools/validation_table.py

paper:          # refreshes the tables from results/ first
	$(PY) tools/paper_tables.py
	cd paper && latexmk -pdf -interaction=nonstopmode abkit.tex && latexmk -c

clean:
	rm -rf build dist *.egg-info .pytest_cache .hypothesis
	find . -name __pycache__ -prune -exec rm -rf {} +
