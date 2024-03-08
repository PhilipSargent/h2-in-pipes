import sys
from pdfreader import SimplePDFViewer

def print_pdf_properties(pdf_path):
    fd = open(pdf_path, 'rb')
    viewer = SimplePDFViewer(fd)

    doc_info = viewer.metadata

    print(f"PDF Information for {pdf_path}:")
    print(f"Author: {doc_info.get('Author')}")
    print(f"Creator: {doc_info.get('Creator')}")
    print(f"Producer: {doc_info.get('Producer')}")
    print(f"Subject: {doc_info.get('Subject')}")
    print(f"Title: {doc_info.get('Title')}")
    print(doc_info)
    fd.close()

# The first command line argument is the PDF file path
print_pdf_properties(sys.argv[1])
