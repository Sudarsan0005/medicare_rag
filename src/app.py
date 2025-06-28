from langchain_community.document_loaders import PyPDFLoader
file_path=r"C:\Users\sudar\Downloads\10050-medicare-and-you_0.pdf"

loader = PyPDFLoader(file_path)
pages = []
for page in loader.load():
    pages.append(page)

print(pages)