import pdfplumber
import csv

def pdf_lines_to_csv(pdf_path, csv_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Page Number', 'Line Number', 'Text'])  # Header row
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line_num, line in enumerate(lines, start=1):
                        if line.strip():  # Skip empty lines
                            writer.writerow([page_num, line_num, line])

if __name__ == '__main__':
    pdf_lines_to_csv('/Users/metasebiya21/personal_project/mezumre_dawit/Mezmuredawit.pdf', 'output.csv')