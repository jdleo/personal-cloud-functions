.PHONY: lint deploy

# To lint Python files
lint:
	@pylint $(shell git ls-files '*.py')

# To deploy the application
deploy:
	@firebase deploy --only functions