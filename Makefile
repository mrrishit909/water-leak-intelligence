export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55210/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8210

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m wa.seed --if-empty
dev:            ## API with reload on :8210, jobs run by a local worker
	($(PY) -m core.jobs wa.api &) && $(PY) -m uvicorn wa.api:app --reload --port 8210
test:
	$(PY) -m pytest --cov=wa --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) utility-viewer-demo 32 10 "GET /v1/network" "GET /v1/leaks" "GET /v1/assets?limit=20" "GET /v1/capital-plan" 
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test reset
