build-docs:
	uv run pdoc -d numpy -o html_docs phoibe

install:
	uv sync --all-extras
	uv run pre-commit install

re-install:
	rm -rf .venv
	$(MAKE) install

test:
	uv run pytest --cov phoibe --cov-report term-missing -m "not rio"
	uv run pytest --cov phoibe --cov-append --cov-report term-missing -m rio

update-precommits:
	uv run pre-commit autoupdate

update-trix-contract-snapshot:
	UPDATE_SNAPESHOTS=1 uv run pytest tests/contract_tests/geography/complexity/pipelines/test_contract_snapshot.py