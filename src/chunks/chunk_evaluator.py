from src.data_manager.data_loader import DataExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm_manager import embeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.constants import (llm_model,base_url)
from src.llm_manager.prompt.prompt_manager import ragEvaluator_prompt
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness,LLMContextPrecisionWithReference
from uuid import uuid4
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO)

SAMPLE_QUESTIONS = [
    "Which plans can people new to Medicare on or after January 1, 2020, buy instead of C and F?",
    "Does Medicare Advantage cover hospice care and clinical trials? If not, who does?",
    "Can you have Medigap and a Medicare Advantage Plan at the same time?",
    "What happens if your primary care provider leaves your Medicare Advantage Plan's network mid-year? Can you change plans?",
    "If a Medicare Advantage Plan doesn’t include Part D drug coverage, can you join a separate drug plan?"
]

SAMPLE_ANSWERS = [
    "Plans D and G",
    "No, Medicare Advantage Plans do not cover hospice care and some costs of clinical trials. These services are still covered by Original Medicare, even if you're enrolled in a Medicare Advantage Plan. Medicare also helps cover benefits that the plan does not provide if they are required by law or Medicare policy.",
                 "No, you cannot have both Medigap and a Medicare Advantage Plan at the same time. If you already have a Medigap policy and you join a Medicare Advantage Plan, you may want to drop Medigap. Also, you can't use Medigap to pay for Medicare Advantage Plan deductibles, copayments, or premiums.",
                 """If your primary care or behavioral health provider leaves your Medicare Advantage Plan’s network mid-year and you’ve seen that provider in the past 3 years, the plan must notify you.
You generally can’t change plans immediately, but:

The plan must help you find a new provider and continue needed care.

You may qualify for a Special Enrollment Period if specific conditions are met.
Read all notices carefully and contact the plan if you have questions about switching.

""",
                 """It depends on the type of Medicare Advantage Plan:

If you're in a Medical Savings Account (MSA) Plan or some Private Fee-for-Service Plans, you can join a separate Medicare drug plan.

If you're in an HMO or PPO that doesn’t include drug coverage, you cannot join a separate drug plan.
In such cases, you would need to use other drug coverage (like from an employer or retiree plan) or go without."""
]


class ChunkSizeEvaluator:
    def __init__(self):
        self.data_extractor = DataExtractor()
        self.embedding_model = embeddings.EmbeddingGenerator()
        self.llm = ChatNVIDIA(model=llm_model,
                              nvidia_base_url=base_url,
                              temperature=0.6
                              )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", ragEvaluator_prompt), ("user", "Question:{question} Context: {context}")]
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        self.index = faiss.IndexFlatL2(len(self.embedding_model.generate("hello world", model_type='nvidia')))
    def evaluate(self, file_path: str, chunk_configs: list):
        try:
            data = self.data_extractor.data_loader(file_path=file_path)
            selected_data = data[61:78]  # Section 4 and 5 focus

            all_results = []

            for config in chunk_configs:
                logging.info(f"Evaluating chunk config: {config}")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config['chunk_size'],
                    chunk_overlap=config['chunk_overlap']
                )
                chunked_data = splitter.split_documents(selected_data)


                uuids = map(str, (uuid4() for _ in range(len(chunked_data))))

                vector_store = FAISS(
                    embedding_function=self.embedding_model.nvidia_model,
                    index=self.index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                vector_store.add_documents(documents=chunked_data, ids=list(uuids))

                dataset = []

                for query, reference in zip(SAMPLE_QUESTIONS, SAMPLE_ANSWERS):
                    retrieved_docs = vector_store.similarity_search(query, k=2)
                    contexts = [doc.page_content for doc in retrieved_docs]
                    response = self.chain.invoke({"question":query,"context":contexts})

                    dataset.append({
                        "user_input": query,
                        "retrieved_contexts": contexts,
                        "response": response,
                        "reference": reference
                    })
                eval_dataset = EvaluationDataset.from_list(dataset)
                result = evaluate(
                    dataset=eval_dataset,
                    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(),LLMContextPrecisionWithReference()],
                    llm=self.evaluator_llm
                )

                all_results.append({
                    "config": config,
                    "result": result
                })
                logging.info(f"Finished evaluation for chunk size {config['chunk_size']}")

            return all_results

        except Exception as e:
            logging.error(f"Evaluation failed: {e}", exc_info=True)
            return None


# Main execution
if __name__ == "__main__":
    evaluator = ChunkSizeEvaluator()
    strategies = [
        {"chunk_size": 512, "chunk_overlap": 50},
        {"chunk_size": 1200, "chunk_overlap": 120},
        {"chunk_size": 1024, "chunk_overlap": 100}
    ]
    results = evaluator.evaluate(file_path=r"C:\Users\sudar\Downloads\10050-medicare-and-you_0.pdf",
                                 chunk_configs=strategies)
    print(results)

    '''
[
  {
    'config': {
      'chunk_size': 512,
      'chunk_overlap': 50
    },
    'result': {
      'context_recall': 0.7733,
      'faithfulness': 0.9500,
      'factual_correctness(mode=f1)': 0.5000,
      'llm_context_precision_with_reference': 1.0000
    }
  },
  {
    'config': {
      'chunk_size': 1200,
      'chunk_overlap': 120
    },
    'result': {
      'context_recall': 1.0000,
      'faithfulness': 0.8833,
      'factual_correctness(mode=f1)': 0.6020,
      'llm_context_precision_with_reference': 1.0000
    }
  },
  {
    'config': {
      'chunk_size': 1024,
      'chunk_overlap': 100
    },
    'result': {
      'context_recall': 0.9333,
      'faithfulness': 0.8500,
      'factual_correctness(mode=f1)': 0.5720,
      'llm_context_precision_with_reference': 1.0000
    }
  }
]'''

## so by observing the metrics result chunk size 700 have better accuracy  compair 500 and 1024
