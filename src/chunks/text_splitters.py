from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
document = """Some documents have an inherent structure, such as HTML, Markdown, or JSON files. In these cases, it's beneficial to split the document based on its structure, as it often naturally groups semantically related text. Key benefits of structure-based splitting:

Preserves the logical organization of the document
Maintains context within each chunk
Can be more effective for downstream tasks like retrieval or summarization
Examples of structure-based splitting:

Markdown: Split based on headers (e.g., #, ##, ###)
HTML: Split using tags
JSON: Split by object or array elements
Code: Split by functions, classes, or logical blocks
Further reading
See the how-to guide for Markdown splitting.
See the how-to guide for Recursive JSON splitting.
See the how-to guide for Code splitting.
See the how-to guide for HTML splitting.
Semantic meaning based
Unlike the previous methods, semantic-based splitting actually considers the content of the text. While other approaches use document or text structure as proxies for semantic meaning, this method directly analyzes the text's semantics. There are several ways to implement this, but conceptually the approach is split text when there are significant changes in text meaning. As an example, we can use a sliding window approach to generate embeddings, and compare the embeddings to find significant differences:"""
texts = text_splitter.split_text(document)
print(texts)

