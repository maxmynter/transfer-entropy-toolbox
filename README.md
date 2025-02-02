# Transfer Entropy Toolbox and application to Futures time series before and during COVID

This is a rigorous rewrite and extension of the research in [my master thesis](https://www.maxmynter.com/master_thesis_mynter.pdf).

This is the directory structure.
```bash
├── README.md
├── analysis
├── scratchpad
├── src
└── tests
```

the `src/te_toolbox` contains all the reusable code.
Mainly this is the entropy functions (univariate, bi-variate, and transfer entropy with different normalizations), utilities to generate synthetic data from chaotic maps, and implementations of various binning methods.

There are two implementations of most of the entropies; one in Python with NumPy optimization and another one in C++ for better performance.
The latter is the default. But you can change the backend to Python like so:
```python
from te_toolbox.entropies.core import import Backend, set_backend

set_backend(Backend.CPP.value)
set_backend(Backend.PY.value)
```

The `analysis` directory contains the `data`, `plots`, and `scripts` directories and the subdirectories for the corresponding experiments.
Data is not included in the version control.
You can recreate the synthetic data by running the corresponding analysis scripts.
The Futures timeseries cannot be shared due to contractual agreement.

The `scratchpad` directory contains code for ad-hoc analysis, benchmarks, and general tinkering that is not rigorous and not part of the paper.

The `tests` directory contains the test suite. That is functional and hypothesis tests as well as tests for consistency with the legacy implementation during my thesis.
More on that below.

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

#### Running Legacy Consistency Tests
I am rewriting large chunks of the codebase that was the foundation of my thesis to incorporate
the practices I've developed as a software engineer since then.

There is a dedicated test suite to test consistencies against the legacy codebase.
The legacy code requires additional dependencies and the tests are disabled by default. To run them:

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
