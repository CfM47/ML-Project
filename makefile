# Makefile for ML-Project
# ---------------------------------
# Usage:
#   make lint       -> Run Ruff linter
#   make lint-fix   -> Fix lint issues automatically
#   make typecheck  -> Run mypy type checking
#   make test       -> Run pytest
#   make cz-check   -> Validate commit message (Commitizen)
#   make cz-branch  -> Validate branch with Commitizen
#   make all        -> Run lint, typecheck, and tests

# Variables
UV = uv

# ---------------------------------
# Linting
lint:
	@echo "Running Ruff..."
	$(UV) run ruff check .
	@echo "---------------"

lint-fix:
	@echo "Running Ruff with fix..."
	$(UV) run ruff check --fix .
	@echo "---------------"

# ---------------------------------
# Type Checking
typecheck:
	@echo "Running Mypy..."
	$(UV) run mypy src/ tests/
	@echo "---------------"

# ---------------------------------
# Testing
test:
	@echo "Running pytest..."
	$(UV) run pytest
	@echo "---------------"

#----------------------------------
# Make commit
commit:
	$(UV) run cz commit

# ---------------------------------
# Run all pre-commit style checks
check_all: lint typecheck test
	@echo "All checks passed!"
