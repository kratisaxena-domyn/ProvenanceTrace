# ProvenanceTrace

A comprehensive toolkit for provenance tracing and claim verification with support for hallucination detection and entity alignment.

## 🚀 Quick Start

### Installation

```bash
# Install the core library
chmod +x setup_dev.sh ./setup_dev.sh

./setup_dev.sh
```

### Basic Usage

```python
from domynclaimalign.main import compute_traces
from domynclaimalign.utils import model_utils

# Example usage
traces = compute_traces.generate_trace(your_data)
```

## 📁 Project Structure

```
ProvenanceTrace/
├── packages/
│   └── domynclaimalign/          # Core Python library
│       ├── main/                 # Main computation modules
│       ├── utils/                # Utility functions
│       └── pyproject.toml        # Package configuration
├── examples/                     # Usage examples and experiments
├── docs/                         # Documentation
└── README.md                     # This file
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Development Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd ProvenanceTrace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e packages/domynclaimalign[dev]

# Install development dependencies
pip install pytest black flake8 mypy
```

### Running Tests

```bash
cd packages/domynclaimalign
python -m pytest tests/
```

## 📚 Core Modules

### Main Components
- **compute_traces.py**: Core tracing computation logic
- **hallucination_claim_support.py**: Hallucination detection and claim support
- **match_entities_in_text.py**: Entity matching and alignment
- **maximal_match_text.py**: Text matching algorithms

### Utilities
- **model_utils.py**: Model loading and management utilities
- **text_matching_utils.py**: Text processing and matching helpers
- **index_utils.py**: Indexing and search utilities
- **json_utils.py**: JSON processing utilities

## 🔧 Configuration

The package uses `pyproject.toml` for configuration. Key settings:
- Dependencies management
- Build system configuration
- Development tools setup

## 📖 Examples

Check the `examples/` directory for:
- Basic usage examples
- Advanced use cases
- Integration examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

See [LICENSE](packages/domynclaimalign/LICENCE) file for details.

## 🆘 Support

- Create an issue for bug reports
- Check existing issues before creating new ones
- Provide detailed information for better support