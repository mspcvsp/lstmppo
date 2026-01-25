# -----------------------------
# Core test targets
# -----------------------------

test:
	pytest -q

cpu:
	pytest tests/cpu -q

gates:
	pytest -m gates -q

drift:
	pytest -m drift -q

policy:
	pytest -m policy -q

ppo:
	pytest -m ppo -q

infra:
	pytest -m infra -q

fast:
	pytest -m fast -q


# -----------------------------
# Linting & formatting
# -----------------------------

lint:
	ruff check .

lint-fix:
	ruff check --fix .

format:
	ruff format .
	black .

style: lint-fix format


# -----------------------------
# CI target (exactly what CI runs)
# -----------------------------

ci: lint test


# -----------------------------
# Help
# -----------------------------

help:
	@echo "Available targets:"
	@echo "  test        - run all tests"
	@echo "  cpu         - run CPU test suite"
	@echo "  gates       - run gate-level invariants"
	@echo "  drift       - run drift & stability invariants"
	@echo "  policy      - run policy-level invariants"
	@echo "  ppo         - run PPO invariants"
	@echo "  infra       - run infrastructure invariants"
	@echo "  fast        - run fast smoke tests"
	@echo "  lint        - run Ruff lint checks"
	@echo "  lint-fix    - auto-fix Ruff issues"
	@echo "  format      - run Ruff + Black formatters"
	@echo "  style       - fix lint + format code"
	@echo "  ci          - run CI pipeline locally"