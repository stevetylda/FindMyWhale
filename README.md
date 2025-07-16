# 🐋 FindMyWhale

> FindMyWhale - Acoustic Localization of Orcas

**FindMyWhale** is a Python toolkit for detecting and localizing orca vocalizations in underwater acoustic recordings. It provides tools to identify whale calls, estimate their spatial position using hydrophone array data, and visualize call trajectories. This project is built for researchers, conservationists, and curious minds interested in underwater bioacoustics.

---

## 🌊 Features

- 🎧 **Call Detection**: Identify orca calls in mono or multichannel audio.
- 📍 **Localization**: Estimate the position of calling whales using time-difference-of-arrival (TDOA) techniques.
- 🌀 **Hydrophone Array Support**: Works with multiple hydrophones for triangulation.
- 📈 **Visualization**: Plot call locations and movement paths in 2D/3D space.
- 🐍 **Pythonic API**: Easily integrate into research workflows or use from the command line.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/FindMyWhale.git
cd FindMyWhale
pip install -r requirements.txt
```

---

## 🧠 Background

In-Progress

---

## 📚 Documentation

Full API documentation coming soon! For now, check out:

examples/
Inline docstrings in findmywhale/

### Repo Organization
```
FindMyWhale/
├── findmywhale/ # Main Python package (your source code)
│ ├── init.py
│ ├── detection.py
│ ├── localization.py
│ ├── visualization.py
│ └── utils.py
│
├── examples/ # Usage examples and demo notebooks
│ └── demo.ipynb
│
├── tests/ # Unit tests (pytest-compatible)
│ ├── init.py
│ ├── test_detection.py
│ ├── test_localization.py
│ └── test_utils.py
│
├── data/ # (Optional) Sample data or audio files
│ └── orca_sample.wav
│
├── README.md # Project overview and usage
├── requirements.txt # Runtime dependencies
├── setup.py # Optional: if you want to make it pip-installable
├── pyproject.toml # Modern build/configuration (optional but recommended)
├── .gitignore # Ignore files like pycache, .env, etc.
└── LICENSE # Your project's license (e.g., MIT)
```
---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, suggestions, or pull requests.

---

## 🐳 Acknowledgments

---

## 🛰️ Future Plans

---