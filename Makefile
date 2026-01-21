build-docs:
	poetry run pdoc -d numpy -o html_docs phoibe

install:
	poetry sync --all-extras --with dev,docs
	poetry run pre-commit install

re-install:
	rm -rf .venv
	$(MAKE) install

test:
	poetry run pytest --cov phoibe --cov-report term-missing

update-precommits:
	poetry run pre-commit autoupdate