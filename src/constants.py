import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
os.environ["NVIDIA_API_KEY"] =os.environ.get("NVIDIA_API_KEY", "")
llm_model = os.environ.get("LLM_MODEL", "")
base_url=os.environ.get("base_url","")
openai_embedding_model = os.environ.get("openai_embedding_model","")
nemo_embedding_model = os.environ.get("nemo_embedding_model","nvidia/nv-embed-v1")

CWD = os.getcwd()

DOC_DIR = os.path.join(CWD, "documents")

