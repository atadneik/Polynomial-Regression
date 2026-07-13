import pypandoc
import os

# Download pandoc if it doesn't exist
print("Downloading pandoc...")
pypandoc.download_pandoc()

# Paths
md_file = r'c:\Users\DELL\Desktop\Polynomial-Regression\EASSAY\Chapter5_Final_EN.md'
docx_file = r'c:\Users\DELL\Desktop\Polynomial-Regression\EASSAY\Chapter5_Final_EN.docx'

print(f"Converting {md_file} to {docx_file}...")
pypandoc.convert_file(md_file, 'docx', outputfile=docx_file)
print("Conversion successful!")
