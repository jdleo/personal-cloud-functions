.PHONY: lint deploy bandit black

# To lint Python files
lint:
	@pylint $(shell git ls-files '*.py')

# To check security
bandit:
	@bandit -r .

# To check formatting
black:
	@black .

# To deploy the application
deploy:
	@firebase deploy --only functions