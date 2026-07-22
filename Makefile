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