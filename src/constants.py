from dotenv import load_dotenv
import getpass
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NVIDIA_API_KEY"] =os.environ.get("NVIDIA_API_KEY", "")
model = os.environ.get("MODEL", "")
base_url=os.environ.get("base_url","")
openai_embedding_model = os.environ.get("embedding_model","")
nemo_embedding_model = "nvidia/nv-embed-v1"
vector_size = 1024