# Procurement Intelligence Platform: one command per stage.
PY := python
export PYTHONPATH := src
export P2P_RAW_DIR := $(CURDIR)/data/raw

.PHONY: help install data warehouse docs experiments artifacts notebooks evals test app all clean

help:            ## Show available commands
	@grep -E '^[a-z]+:.*##' Makefile | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:         ## Install the full laboratory (all models, notebooks, tests)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

data:            ## Generate the fictional SAP extracts and the ground truth (about 10 s)
	$(PY) -m p2p.generator.export

warehouse:       ## Build and test the dbt warehouse (33 models, 42 tests)
	cd warehouse && dbt build --profiles-dir .

docs:            ## Generate the dbt documentation site
	cd warehouse && dbt docs generate --profiles-dir . --static

experiments:     ## Run every backtest of the model arena (about 30 minutes on 2 CPUs; cached afterwards)
	$(PY) scripts/run_experiments.py

notebooks:       ## Execute all notebooks in place
	cd notebooks && for nb in *.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $$nb; done

artifacts:       ## Build forecasts, ETAs and the compact database used by the app and the agent
	$(PY) scripts/build_artifacts.py

evals:           ## Evaluate the AI agent (offline mode unless an API key is set)
	$(PY) evals/run_evals.py

test:            ## Run the test suite
	$(PY) -m pytest -q

app:             ## Launch the dashboard on http://localhost:8501
	streamlit run app/streamlit_app.py

all: data warehouse experiments notebooks artifacts evals test  ## Rebuild everything from scratch

clean:           ## Remove generated data (keeps the app artifacts)
	rm -rf data/raw data/ground_truth data/warehouse warehouse/target
