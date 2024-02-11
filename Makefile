.PHONY: lint deploy bandit black check

# To lint Python files
lint:
	@pylint $(shell git ls-files '*.py')

# To check security
bandit:
	@bandit -lll -r -x "./functions/venv/*" .

# To check formatting
black:
	@black .

# To deploy the application
deploy:
	@firebase deploy --only functions --token "${FIREBASE_TOKEN}"

# To check everything
check: lint bandit black
	@echo "All checks passed ✅"