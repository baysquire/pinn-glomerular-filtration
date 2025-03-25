# Contributing

Thank you for your interest in contributing to this project.

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to your branch (`git push origin feature/your-feature`)
5. **Open a Pull Request**

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 for Python code
- Include docstrings for all functions and classes
- Keep physics and neural network logic separated (see `src/`)

## Reporting Issues

Please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
