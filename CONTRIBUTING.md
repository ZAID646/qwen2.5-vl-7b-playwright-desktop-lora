# Contributing

Thank you for your interest in contributing to the Qwen2.5-VL-7B Playwright Desktop LoRA project.

## How to Contribute

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Make your changes.
4. Run the test suite:
   ```bash
   pytest
   ```
5. Lint with ruff:
   ```bash
   ruff check .
   ```
6. Commit with a conventional commit message:
   - `feat:` — new feature
   - `fix:` — bug fix
   - `docs:` — documentation only
   - `refactor:` — code restructuring
   - `test:` — test additions or changes
   - `chore:` — maintenance tasks
7. Push to your fork and open a pull request.

## Code Style

- Python 3.11+ with `from __future__ import annotations`
- Type hints on all public functions and methods
- 100-character line limit
- Ruff and mypy strict mode enforced via `pyproject.toml`

## Pull Request Guidelines

- Keep PRs focused on a single concern.
- Include tests for new functionality.
- Update documentation if public APIs change.
- Ensure all CI checks pass.

## Reporting Issues

Report bugs or request features by opening a GitHub Issue. Include:
- A clear title and description
- Steps to reproduce (for bugs)
- Environment details (OS, Python version, CUDA version if applicable)
