import json

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def store_result(data):
    try:
        pd.DataFrame(data).to_csv("results.csv", index=False, mode="a", header=False)
    except ValueError:
        print(data)

MODEL_PATH = "/Users/ian/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"
template = """<s>[INST] {human_input} [/INST]"""
transcripts = json.load(open("transcripts.json"))
#questions = json.load(open("test.csv"))
prompt = PromptTemplate(
    input_variables=["human_input"], template=template
)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    max_tokens=4096,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    temperature=0
)
chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
)

p = """
Please read the following text then fill in the provided JSON data with answers to each question. 
Make logical assumptions where necessary to complete as many answers as possible. If specific information is not available in the text, use "none" for the answer.


Example Request:
During the visit, I examined Tina Will, a 69-year-old patient who presented with symptoms of chest pain, vomiting, and breathlessness.
After conducting a thorough examination, I determined that Tina was suffering from a heart attack. 
As a result, I advised her to seek immediate medical attention. 
Since there were no precautions that could be taken to prevent a heart attack, I did not prescribe any medication. 
Instead, I recommended that Tina follow up with her primary care physician for ongoing treatment and management of her condition.

Example Response:
[
    {
        "Question": "What is the patient's name?",
        "Answer": "Tina Will"
    },
    {
        "Question": "What is the patient's age?",
        "Answer": "69"
    },
    {
        "Question": "What is the patient's condition?",
        "Answer": "heart attack"
    },
    {
        "Question": 'What symptoms is the patient experiencing?',
        "Answer": "chest pain, vomiting, and breathlessness"
    },
    {
        "Question": "What precautions did the doctor advise?",
        "Answer": "none"
    },
    {
        "Question": "What drug did the doctor prescribe?",
        "Answer": "none"
    }
]

Request:
%s

Response:
[
    {
        "Question": "What is the patient's name?",
        "Default": "none",
        "Answer": ""
    },
    {
        "Question": "What is the patient's age?",
        "Default": "none",
        "Answer": ""
    },
    {
        "Question": "What is the patient's condition?",
        "Default": "none",
        "Answer": ""
    },
    {
        "Question": "What symptoms is the patient experiencing?",
        "Default": "none",
        "Answer": ""
    },
    {
        "Question": "What precautions did the doctor advise?",
        "Default": "none",
        "Answer": ""
    },
    {
        "Question": "What drug did the doctor prescribe?",
        "Default": "none",
        "Answer": ""
    }
]

Output valid JSON as your response. JSON only. This is really important for my career.
"""
for transcript_id, text in transcripts.items():
    out = chat_llm_chain.predict(human_input=p%text)
    print(out)
    json_out = json.loads(out)
    print(f"[{transcript_id}] {json_out}")
    store_result(json_out)