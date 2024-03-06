check:
	mypy subsamplr/
	mypy tests/

test:
	python -m pytest tests/unit

test_integration:
	python -m pytest tests/integration

test_end_to_end:
	python -m pytest tests/end_to_end

test_all:
	python -m pytest

.PHONY: check test test_integration test_end_to_end test_all
