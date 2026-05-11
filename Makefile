SHELL := /bin/bash
VENV_PY := $(CURDIR)/yuqueai/bin/python

.PHONY: dev down status build-ui run test

dev:
	./scripts/dev_up.sh

down:
	./scripts/dev_down.sh

status:
	./scripts/dev_status.sh

build-ui:
	./scripts/build_frontend.sh

run:
	$(VENV_PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	source scripts/activate.sh && pytest
