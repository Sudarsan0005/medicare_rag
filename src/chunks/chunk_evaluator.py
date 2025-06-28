# import os
# import time
# from llama_index.core import (
#     SimpleDirectoryReader, VectorStoreIndex, ServiceContext,
#     Document, StorageContext, Settings, load_index_from_storage
# )
# from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, ResponseEvaluator
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# # Load environment variables for OpenAI key
# # from dotenv import load_dotenv
# #
# # load_dotenv()
# #
# # # Set LlamaIndex LLM and embedding
# # Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
# # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
#
# # Test queries
# test_queries = [
#     "What is the company’s vision and mission?",
#     "Summarize the regulatory section of the document.",
#     "What are the key points discussed in the risk assessment?"
# ]
#
# # Evaluation results will be stored here
# evaluation_results = []
#
# # Define different chunk sizes and overlaps
# chunking_strategies = [
#     {"chunk_size": 256, "chunk_overlap": 30},
#     {"chunk_size": 512, "chunk_overlap": 50},
#     {"chunk_size": 1024, "chunk_overlap": 100}
# ]
#
# # Path to documents
# docs = SimpleDirectoryReader("data").load_data()
#
# for strategy in chunking_strategies:
#     print(f"\nEvaluating strategy: Chunk size={strategy['chunk_size']} | Overlap={strategy['chunk_overlap']}")
#
#     # LangChain Text Splitter
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=strategy['chunk_size'],
#         chunk_overlap=strategy['chunk_overlap']
#     )
#
#     texts = splitter.split_documents([Document(text=d.text) for d in docs])
#
#     # Convert back to LlamaIndex Document
#     split_docs = [Document(text=doc.page_content) for doc in texts]
#
#     # # Build Index
#     # index = VectorStoreIndex.from_documents(split_docs)
#     # query_engine = index.as_query_engine()
#     #
#     # # Evaluators
#     # faithfulness_evaluator = FaithfulnessEvaluator()
#     # relevancy_evaluator = RelevancyEvaluator()
#     #
#     # total_time = 0
#     # total_faithfulness = 0
#     # total_relevancy = 0
#     #
#     # for query in test_queries:
#     #     start = time.time()
#     #     response = query_engine.query(query)
#     #     end = time.time()
#     #
#     #     # Evaluate
#     #     faithfulness = faithfulness_evaluator.evaluate_response(query=query, response=str(response))
#     #     relevancy = relevancy_evaluator.evaluate_response(query=query, response=str(response))
#     #
#     #     print(f"\nQuery: {query}")
#     #     print(f"Response: {response}")
#     #     print(f"Faithfulness Score: {faithfulness.score}")
#     #     print(f"Relevancy Score: {relevancy.score}")
#     #     print(f"Response Time: {end - start:.2f} seconds")
#     #
#     #     total_time += (end - start)
#     #     total_faithfulness += faithfulness.score
#     #     total_relevancy += relevancy.score
#     #
#     # avg_time = total_time / len(test_queries)
#     # avg_faithfulness = total_faithfulness / len(test_queries)
#     # avg_relevancy = total_relevancy / len(test_queries)
#     #
#     # evaluation_results.append({
#     #     "chunk_size": strategy['chunk_size'],
#     #     "chunk_overlap": strategy['chunk_overlap'],
#     #     "avg_time": avg_time,
#     #     "avg_faithfulness": avg_faithfulness,
#     #     "avg_relevancy": avg_relevancy
#     # })
#
# # Show Summary
# print("\n====== Summary of Chunking Strategy Evaluation ======")
# for result in evaluation_results:
#     print(f"Chunk Size: {result['chunk_size']} | Overlap: {result['chunk_overlap']}")
#     print(f" - Avg Time: {result['avg_time']:.2f}s")
#     print(f" - Avg Faithfulness: {result['avg_faithfulness']:.2f}")
#     print(f" - Avg Relevancy: {result['avg_relevancy']:.2f}\n")

from src.data_manager.data_loader import DataExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm_manager import (nvidia_llm,embeddings)
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, ServiceContext,
    Document, StorageContext, Settings, load_index_from_storage
)
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    context_entity_recall
)
from ragas.metrics.critique import harmfulness
from ragas.integrations.llama_index import evaluate

class ChunkSizeEvalutor:
    def __init__(self):
        self.data_extractor = DataExtractor()
        self.embedding_mod=embeddings.EmbeddingGenerator().nvidia_model
        self.llm = nvidia_llm.LlmManager(model="meta/llama-3.3-70b-instruct").llm
    def evaluate(self,file_path:str,chunk_config:list):
        try:
            data = self.data_extractor.data_loader(file_path=file_path)
            Settings.llm = self.llm
            Settings.embed_model = self.embedding_mod
            data=data[61:78] # setting experiment on section 4 and 5 in document
            for strategy in chunk_config:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=strategy['chunk_size'],
                    chunk_overlap=strategy['chunk_overlap']
                )
                chunk_data = splitter.split_documents(data)
                split_docs = [Document(text=doc.page_content,metadata=doc.metadata) for doc in chunk_data]
                # Build Index
                index = VectorStoreIndex.from_documents(split_docs)
                query_engine = index.as_query_engine(similarity_top_k=2)

                # Evaluators
                faithfulness_evaluator = FaithfulnessEvaluator()
                relevancy_evaluator = RelevancyEvaluator()

                total_time = 0
                total_faithfulness = 0
                total_relevancy = 0

                for query in test_queries:
                    start = time.time()
                    response = query_engine.query(query)
                    end = time.time()

                    # Evaluate
                    faithfulness = faithfulness_evaluator.evaluate_response(query=query, response=str(response))
                    relevancy = relevancy_evaluator.evaluate_response(query=query, response=str(response))

                    print(f"\nQuery: {query}")
                    print(f"Response: {response}")
                    print(f"Faithfulness Score: {faithfulness.score}")
                    print(f"Relevancy Score: {relevancy.score}")
                    print(f"Response Time: {end - start:.2f} seconds")

                    total_time += (end - start)
                    total_faithfulness += faithfulness.score
                    total_relevancy += relevancy.score

                avg_time = total_time / len(test_queries)
                avg_faithfulness = total_faithfulness / len(test_queries)
                avg_relevancy = total_relevancy / len(test_queries)

                evaluation_results.append({
                    "chunk_size": strategy['chunk_size'],
                    "chunk_overlap": strategy['chunk_overlap'],
                    "avg_time": avg_time,
                    "avg_faithfulness": avg_faithfulness,
                    "avg_relevancy": avg_relevancy
                })

            # Show Summary
            print("\n====== Summary of Chunking Strategy Evaluation ======")
            for result in evaluation_results:
                print(f"Chunk Size: {result['chunk_size']} | Overlap: {result['chunk_overlap']}")
                print(f" - Avg Time: {result['avg_time']:.2f}s")
                print(f" - Avg Faithfulness: {result['avg_faithfulness']:.2f}")
                print(f" - Avg Relevancy: {result['avg_relevancy']:.2f}\n")
        except Exception as e:
            pass

chnk_evl = ChunkSizeEvalutor()
chunking_strategies = [
    # {"chunk_size": 256, "chunk_overlap": 30},
    {"chunk_size": 512, "chunk_overlap": 50},
    # {"chunk_size": 1024, "chunk_overlap": 100}
]
chnk_evl.evaluate(file_path=r"C:\Users\sudar\Downloads\10050-medicare-and-you_0.pdf",chunk_config=chunking_strategies)