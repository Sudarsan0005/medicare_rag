from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.constants import (base_url)


class LlmManager:
    def  __init__(self,model,sys_prompt:str=None):
        self.llm = ChatNVIDIA(model=model,
                              nvidia_base_url=base_url,
                              temperature=0.3
                              )
        self.prompt_template=ChatPromptTemplate.from_messages(
    [("system",sys_prompt ), ("user", "Question:{question} Context: {context}")]
)
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    async def chat_llm(self,question:str=None,context:any=None)->str:
        try:
            response=self.chain.invoke({"question":question,"context":context})
            return response
        except Exception as e:
            raise Exception(f"llm fail to generate {e}")