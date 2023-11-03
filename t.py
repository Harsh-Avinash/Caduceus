from langchain.llms import GPT4All, LlamaCpp
import os
from dotenv import load_dotenv

load_dotenv()

model_path = "models/ggml-vic7b-uncensored-q5_1"
model_n_ctx = os.getenv('MODEL_N_CTX')
model_n_batch = int(os.getenv('MODEL_N_BATCH', 8))


def ask_llm(llm, prompt):
    # Convert the prompt into a list with a single element
    response = llm.generate([prompt])
    return response


# Usage:
llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='ggml',
              n_batch=model_n_batch, verbose=False)  # type: ignore

prompt = "Write a song about how society is doomed beautiful"
response = ask_llm(llm, prompt)

print(response)
