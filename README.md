# Transfer Entropy Toolbox

A paper from my M.Sc. thesis about the problems of Transfer Entropy from discretized continuous time series.

## Development Setup

### Installation
Clone the repository and install in development mode with:
```bash
git clone https://github.com/maxmynte/msc-paper.git
cd te-toolbox
pip install -e ".[dev]"
```

### Testing
The project uses pytest and hypothesis for testing. The test suite is split into two parts:
1. Regular tests for the current implementation
2. Legacy tests that check consistency with the original thesis implementation

#### Running Regular Tests
By default, only the regular test suite runs. Install development dependencies and run:
```bash
pip install -e ".[dev]"
pytest
```

#### Running Legacy Tests
The legacy tests require additional dependencies and are disabled by default. To run them:

1. Install legacy test dependencies:
```bash
pip install -e ".[legacy-tests]"
```

2. Run tests with one of these options:
```bash
# Run only legacy tests
pytest -m legacy

# Run all tests including legacy
pytest -m "legacy or not legacy"
```
