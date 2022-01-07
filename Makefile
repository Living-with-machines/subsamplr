check:
	mypy subsamplr/
	mypy tests/
	python -m pytest tests/unit

test_integration:
	python -m pytest tests/integration

test_all:
	python -m pytest

.PHONY: check test_integration test_all
