# GitHub Actions Workflows Status

This document provides an overview of all GitHub Actions workflows in the CRISP Python project.

## Workflow Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/sigridjineth/crisp-py/actions/workflows/ci.yml/badge.svg)](https://github.com/sigridjineth/crisp-py/actions/workflows/ci.yml)
[![Format Check](https://github.com/sigridjineth/crisp-py/actions/workflows/format-check.yml/badge.svg)](https://github.com/sigridjineth/crisp-py/actions/workflows/format-check.yml)
[![Type Check](https://github.com/sigridjineth/crisp-py/actions/workflows/type-check.yml/badge.svg)](https://github.com/sigridjineth/crisp-py/actions/workflows/type-check.yml)
[![codecov](https://codecov.io/gh/sigridjineth/crisp-py/branch/main/graph/badge.svg)](https://codecov.io/gh/sigridjineth/crisp-py)
```

## Workflows Overview

### 1. CI (ci.yml)
- **Trigger**: Push to main/develop, PRs, manual
- **Jobs**:
  - Test (Python 3.11, 3.12) with coverage
  - Lint (flake8, black check)
  - Type check (mypy)
  - Security scan (bandit, safety)
  - Build distribution
- **Features**:
  - Matrix testing across Python versions
  - Coverage reports with codecov integration
  - Artifact uploads
  - Fast failure on errors

### 2. Format Check (format-check.yml)
- **Trigger**: Push to main/develop, PRs, manual
- **Jobs**:
  - Check black formatting
  - Check import sorting (isort)
  - Run flake8 linting
  - Auto-format on PRs (optional)
- **Features**:
  - Detailed formatting diffs in summary
  - Auto-commit formatting fixes on PRs
  - Clear error messages

### 3. Type Check (type-check.yml)
- **Trigger**: Push/PR on Python files only
- **Jobs**:
  - MyPy type checking (Python 3.11, 3.12)
  - Pyright checking (optional/informational)
- **Features**:
  - File annotations for errors
  - HTML reports
  - Caching for faster runs

### 4. Dependency Update (dependency-update.yml)
- **Trigger**: Weekly schedule, manual with strategy selection
- **Jobs**:
  - Update dependencies based on strategy
  - Run tests after update
  - Create PR with changes
- **Features**:
  - Multiple update strategies (patch, minor, major, latest)
  - Automatic PR creation
  - Test validation

### 5. Release (release.yml)
- **Trigger**: Version tags (v*.*.*), manual
- **Jobs**:
  - Version validation
  - Build and test
  - Publish to PyPI
  - Create GitHub release
- **Features**:
  - Automatic changelog generation
  - Pre-release support
  - PyPI publishing

### 6. Benchmark (benchmark.yml)
- **Trigger**: Push to main, PRs with Python changes
- **Jobs**:
  - Performance benchmarks
  - Memory profiling
  - CodSpeed integration
- **Features**:
  - Historical performance tracking
  - PR comments with results
  - Alert on performance regression

## Setup Instructions

1. **Required Secrets**:
   - `PYPI_API_TOKEN`: For publishing to PyPI
   - `CODECOV_TOKEN`: For coverage reports (optional)
   - `CODSPEED_TOKEN`: For performance tracking (optional)

2. **Required Files**:
   - `pyproject.toml`: Project configuration
   - `uv.lock`: Dependency lock file
   - `Makefile`: With test, lint, format, type-check targets

3. **Optional Configuration**:
   - `.flake8`: Flake8 configuration
   - `.black`: Black configuration
   - `mypy.ini` or `pyproject.toml`: MyPy configuration
   - `.coverage.rc` or `pyproject.toml`: Coverage configuration

## Best Practices

1. **Branch Protection**: Enable branch protection rules for main/develop
2. **Required Checks**: Make CI, format-check, and type-check required
3. **Auto-merge**: Enable auto-merge for Dependabot PRs that pass checks
4. **Caching**: Workflows use uv caching for faster runs
5. **Artifacts**: Test results and reports are uploaded as artifacts

## Troubleshooting

### Common Issues

1. **uv not found**: Ensure `astral-sh/setup-uv@v3` action is used
2. **Import errors**: Check that `uv sync` includes all extras
3. **Type errors**: Ensure type stubs are installed for dependencies
4. **Coverage missing**: Install `pytest-cov` and configure properly

### Performance Tips

1. Use path filters to avoid unnecessary runs
2. Cache dependencies aggressively
3. Run jobs in parallel when possible
4. Use fail-fast strategy for matrix builds

## Maintenance

- Review and update workflow dependencies monthly
- Monitor workflow run times and optimize if needed
- Keep Python versions in sync with project requirements
- Update action versions when security updates are available