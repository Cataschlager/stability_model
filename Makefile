.PHONY: all install ingest features model validate test serve report clean help

# Default target
all: ingest features model validate report
	@echo "✅ Full pipeline complete."

help:
	@echo "Spectral Instability Model — Available targets:"
	@echo ""
	@echo "  make install     — Install dependencies via uv"
	@echo "  make ingest      — Download and normalize all data sources"
	@echo "  make features    — Build indicator matrix (impute, standardize, assign pillars)"
	@echo "  make model       — Run PCA, build coupling matrix, calibrate dynamics"
	@echo "  make validate    — Historical reconstruction, rank correlation, AUC"
	@echo "  make test        — Run pytest suite"
	@echo "  make serve       — Launch Streamlit simulation dashboard"
	@echo "  make report      — Generate REPORT.pdf"
	@echo "  make all         — Full pipeline (ingest → features → model → validate → report)"
	@echo "  make check-env   — Validate .env file and test API connections"
	@echo "  make clean       — Remove generated data (keeps raw downloads)"
	@echo ""

install:
	uv sync
	@echo "✅ Dependencies installed."

check-env:
	uv run python -c "from dotenv import load_dotenv; load_dotenv(); import os; \
		keys = ['ACLED_EMAIL','ACLED_API_KEY','UN_POP_BEARER_TOKEN']; \
		missing = [k for k in keys if not os.getenv(k)]; \
		print('✅ All required keys set.' if not missing else f'❌ Missing: {missing}')"

ingest:
	uv run python -m ingestion.run_all
	@echo "✅ Data ingestion complete."

features:
	uv run python -m features.build_indicators
	@echo "✅ Indicator matrix built."

model:
	uv run python -m model.run_all
	@echo "✅ Spectral model complete."

validate:
	uv run python -m validation.run_all
	@echo "✅ Validation complete."

test:
	uv run pytest tests/ -v
	@echo "✅ Tests passed."

serve:
	uv run streamlit run app/streamlit_app.py --server.port 8501

report:
	uv run python notebooks/figures.py
	@echo "✅ Report figures generated."

clean:
	rm -rf data/clean/*
	rm -rf data/output/*
	@echo "🧹 Cleaned generated data. Raw downloads preserved."
