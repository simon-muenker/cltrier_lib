.PHONY: lint
lint:
	@poetry run ruff check --fix
	@poetry run ruff format


.PHONY: deploy
deploy:
	@poetry build
	@poetry publish