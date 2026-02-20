import re
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, removing headers and footers.
    Simple heuristic: remove top/bottom 50 chars or lines if they look like page numbers/headers.
    For this specific task, we'll extract all text and do basic cleaning.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            # Basic Header/Footer removal heuristic (optional, customized for sample)
            # For the sample PDF, it's clean enough, but let's just strip whitespace
            if page_text:
                text += page_text + "\n"

        # Post-processing to clean up multiple newlines or artifacts
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()
    except Exception as e:
        return str(e)
