# ProvenanceTrace

A comprehensive toolkit for provenance tracing and claim verification with support for hallucination detection and entity alignment.

## 📁 Project Structure

```
ProvenanceTrace/
├── packages/
│   └── domynclaimalign/          # Core Python library
│       ├── main/                 # Main computation modules
│       ├── utils/                # Utility functions
│       └── pyproject.toml        # Package configuration
├── examples/                     # Usage examples and experiments
│   ├── FAV/                      # FAV example
│   └── OWFA/                     # OWFA example
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
git clone <this-repository>
cd ProvenanceTrace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e packages/domynclaimalign[dev]
```
Or try:

### Installation

```bash
# Install the core library
chmod +x setup_dev.sh ./setup_dev.sh

./setup_dev.sh
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
- Basic usage examples using large context data: OWFA
- Advanced use cases using financial agentic AI: FAV

## 📄 License

See [LICENSE](LICENCE) file for details.
