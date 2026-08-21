"""
inspect_pdfs.py
----------------
Diagnostic script: prints raw extracted text for a handful of pages from
each new PDF, plus a scan for lines that LOOK like they could be section
headers (short lines, mostly capitalized, contain digits, etc).

Run this from the same folder/venv as your ingest.py:

    python inspect_pdfs.py

Edit FILES below to point at your two new PDFs.
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# ---- EDIT THESE PATHS ----
FILES = [
    "data/Endometriosis_is_a_chronic_disease_affecting_women_of_reproductive.pdf",
    "data/ESHRE GUIDELINE ENDOMETRIOSIS 2022_2.pdf",
]
# Which pages to dump in full (0-indexed). Keep small -- this is just for eyeballing format.
SAMPLE_PAGES = [0, 1, 2, 5, 10]
# ---------------------------


def looks_like_possible_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    # crude heuristics: starts with a digit/roman numeral/keyword,
    # OR is short + mostly uppercase, OR contains "Question"/"Recommendation"
    if any(s.upper().startswith(k) for k in (
        "RECOMMENDATION", "QUESTION", "SECTION", "CHAPTER",
        "ANNEX", "GPP", "GOOD PRACTICE"
    )):
        return True
    if s[0].isdigit():
        return True
    if len(s) < 80 and s == s.upper() and any(c.isalpha() for c in s):
        return True
    return False


def main():
    for path_str in FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"!! Not found: {path_str} (fix the path in FILES)")
            continue

        print("\n" + "=" * 80)
        print(f"FILE: {path.name}")
        print("=" * 80)

        loader = PyPDFLoader(str(path))
        pages = loader.load()
        print(f"Total pages extracted: {len(pages)}\n")

        # 1) Full raw dump of a few sample pages
        for idx in SAMPLE_PAGES:
            if idx >= len(pages):
                continue
            print(f"\n--- RAW TEXT: page index {idx} (page_number meta={pages[idx].metadata.get('page')}) ---")
            print(pages[idx].page_content[:1500])
            print("--- end sample ---")

        # 2) Scan ALL pages for lines that might be headers
        print("\n--- CANDIDATE HEADER LINES (heuristic scan, all pages) ---")
        count = 0
        for p_idx, page in enumerate(pages):
            for line in page.page_content.splitlines():
                if looks_like_possible_header(line):
                    print(f"  p{p_idx}: {line.strip()[:100]}")
                    count += 1
                    if count >= 80:
                        print("  ... (truncated at 80 matches)")
                        break
            if count >= 80:
                break

        print(f"\nTotal candidate header lines found: {count}")


if __name__ == "__main__":
    main()
