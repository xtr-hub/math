# math

Data science project.

## Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Structure

```
math/
├── notebooks/   # Exploratory analysis (01-XX naming)
├── data/
│   ├── raw/     # Immutable original data
│   └── processed/
├── src/         # Reusable modules
├── outputs/     # Figures, models, reports
├── tests/       # Unit tests
└── pyproject.toml
```
