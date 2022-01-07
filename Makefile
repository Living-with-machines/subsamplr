check:
	mypy subsamplr/
	mypy tests/
	python -m pytest

.PHONY: check
