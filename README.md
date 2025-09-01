# Coherence Figures (Reproduction Scripts)

This repository contains a single Python script to reproduce all figures used in the manuscript:

**"Coherence Freezing and Tunable Decoherence-Free Subspaces in Collectively Decaying Qutrits"**  
*R. Sufiani, S. Saei*  
Faculty of Physics, Department of Theoretical Physics and Astrophysics,  
University of Tabriz, Tabriz 51664, Iran  

The script generates all figures in PDF format (300 dpi).  
It relies only on **NumPy** and **Matplotlib** and does not require LaTeX.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
