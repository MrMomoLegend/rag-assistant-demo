"""
extract_corpus.py: convert the FYP's document corpus to plain text, once.

Uses the same extraction as the FYP's main.extract_text (pages joined with a
blank line, then stripped) and writes UTF-8 with LF line endings, so chunking
the .txt gives exactly the chunks the FYP got from the PDFs (1,551 with the
README). The script checks that for every file.

Usage:
    python scripts/extract_corpus.py --src C:/Users/mokit/ai-orchestrator/docs --out corpus
"""
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
)


def chunk(text):
    text = text.strip()
    return [c for c in SPLITTER.split_text(text) if c.strip()] if text else []


def extract(path):
    if path.suffix.lower() == ".pdf":
        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
        return "\n\n".join(pages).strip()
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the FYP's docs/ folder (read only)")
    ap.add_argument("--out", default="corpus")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED)

    total = 0
    for p in files:
        text = extract(p)
        target = out / (p.stem + ".txt" if p.suffix.lower() == ".pdf" else p.name)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        direct = chunk(text)
        via_txt = chunk(target.read_text(encoding="utf-8", errors="replace"))
        assert direct == via_txt, f"chunking changed for {p.name}"
        total += len(via_txt)
        print(f"{p.name}: {len(via_txt)} chunks")

    print(f"\n{len(files)} files, {total} chunks (expected 23 files, 1551 chunks)")
    if total != 1551:
        raise SystemExit("Chunk count differs from the FYP. Check pypdf and langchain-text-splitters versions.")


if __name__ == "__main__":
    main()
