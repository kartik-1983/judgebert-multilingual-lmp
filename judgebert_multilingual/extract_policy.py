from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("../data/verified_sources/pdfs/future_generali_sfsp.pdf")
output_path = Path("../data/verified_sources/future_generali_extracted.txt")

reader = PdfReader(str(pdf_path))

print("PDF pages:", len(reader.pages))

with output_path.open("w", encoding="utf-8") as f:
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        f.write(f"\n\n===== PAGE {page_num} =====\n")
        f.write(text)

print("Extraction complete:", output_path.resolve())
print("Extracted characters:", output_path.stat().st_size)
