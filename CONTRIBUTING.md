# Contributing to Hydra MLOps Framework

Thank you for considering contributing to the Hydra MLOps Framework! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

Bug reports help us improve the framework. When you report a bug, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, LanceDB version, etc.)

Please use the bug report template when opening an issue.

### Suggesting Enhancements

Enhancement suggestions are always welcome. When suggesting an enhancement, please include:

- A clear, descriptive title
- Detailed description of the proposed enhancement
- Any relevant examples or mock-ups
- Explanation of why this enhancement would be useful

Please use the feature request template when opening an issue.

### Pull Requests

We welcome pull requests from the community. To submit a pull request:

1. Fork the repository
2. Create a new branch from `main`: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes with descriptive commit messages
6. Push your branch to your fork
7. Open a pull request against our `main` branch

### Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable and function names
- Write docstrings for all functions, classes, and modules
- Add type hints where appropriate
- Add unit tests for new functionality

## Development Environment Setup

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hydra-mlops.git
cd hydra-mlops
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

We use pytest for testing. To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=.
```

## Project Structure

The project is organized into the following directories:

- `hydra_mlops/api/`: FastAPI application
- `hydra_mlops/connectors/`: Data connector modules
- `hydra_mlops/processors/`: Feature processor modules
- `hydra_mlops/trainers/`: Model trainer modules
- `hydra_mlops/registry/`: Model registry modules
- `hydra_mlops/serving/`: Model serving modules
- `hydra_mlops/monitoring/`: Model monitoring modules
- `hydra_mlops/workflows/`: n8n.io workflow JSON files
- `tests/`: Test suite

## Adding New Components

### Adding a New Connector

1. Create a new file in the `hydra_mlops/connectors/` directory
2. Implement the `DataConnector` interface
3. Register your connector in `hydra_mlops/connectors/__init__.py`
4. Add tests in `tests/connectors/`

### Adding a New Trainer

1. Create a new file in the `hydra_mlops/trainers/` directory
2. Implement the `ModelTrainer` interface
3. Register your trainer in `hydra_mlops/trainers/__init__.py`
4. Add tests in `tests/trainers/`

### Adding a New Workflow

1. Create a new workflow in n8n.io
2. Export the workflow as JSON
3. Save the JSON file in the `hydra_mlops/workflows/` directory
4. Update the documentation to reference the new workflow

## Documentation

When adding new features, please update the documentation:

1. Update the relevant README sections
2. Add or update docstrings
3. Update the API documentation if applicable

## Release Process

1. Ensure all tests pass
2. Update the version number in `setup.py`
3. Update the CHANGELOG.md
4. Create a new GitHub release with a tag matching the version number

## Getting Help

If you need help with the contribution process or have questions, please:

- Open a discussion on GitHub
- Reach out to the maintainers

Thank you for contributing to the Hydra MLOps Framework!
