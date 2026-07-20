"""Train + calibrate the registry-first, 5-year risk model and build the guideline RAG index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neh import risk, guidelines

if __name__ == "__main__":
    risk.train_and_save()
    n = guidelines.build_index()
    print(f"[guidelines] RAG index built with {n} passages")
