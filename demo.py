import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("data/cuad/full_contract_txt")
DEST_DIR = Path("demo_contracts")
NUM_CONTRACTS = 40

random.seed(42)


def main():
    DEST_DIR.mkdir(exist_ok=True)
    all_files = sorted(SOURCE_DIR.glob("*.txt"))
    if not all_files:
        raise SystemExit(f"No .txt files found in {SOURCE_DIR} — check the path.")

    chosen = random.sample(all_files, min(NUM_CONTRACTS, len(all_files)))
    for f in chosen:
        shutil.copy(f, DEST_DIR / f.name)

    print(f"Copied {len(chosen)} contracts into {DEST_DIR}/")
    print("These are small text files and safe to commit to git.")


if __name__ == "__main__":
    main()