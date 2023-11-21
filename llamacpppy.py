import json
from pprint import pprint

from llama_cpp import Llama
import pandas as pd
from langdetect import detect


MODEL_PATH = "/Users/ian/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf"
transcripts = json.load(open("transcripts.json"))
all_questions = pd.read_csv("questions.csv").to_dict(orient="records")

def store_result(data):
    try:
        pd.DataFrame(data).to_csv("results.csv", index=False, mode="a", header=False)
    except ValueError:
        print(data)

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    verbose=False,
)

p = """
<s>[INST] 
Please read the following text then fill in the provided JSON data with answers to each question. 
Make logical assumptions where necessary to complete as many answers as possible. 
If specific information is not available in the text, use "none" for the answer.

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

Do not modify the JSON structure. Output valid JSON as your response.
 [/INST]
"""

translate_prompt = """
<s>[INST]
Please translate the following text into English.

%s

[/INST]
"""

def predict(prompt_template, text):
    formatted_prompt = prompt_template%text
    output = llm(formatted_prompt, max_tokens=4096, echo=False, stop=["</s>"], temperature=0.0)
    pprint(output)
    res_str = output['choices'][0]['text'].replace(formatted_prompt, '')
    return res_str

def translate(text):
    return predict(translate_prompt, (text,))

def extract(text):
    return predict(p, (text,))

start_after = 1033
start = False if start_after else True
for transcript_id, text in transcripts.items():
    transcript_id = int(transcript_id)
    if transcript_id == start_after:
        start = True
        continue
    if not start:
        continue
    # detect language
    language = detect(text)
    if language != "en":
        print(f"Language is {language}. Translating {transcript_id}")
        text = translate(text)
    res_str = extract(text)
    print(res_str)
    json_out = json.loads(res_str)

    questions = [x for x in all_questions if x["Transcript"] == transcript_id]
    json_out = [{"question_id": q['Id'], "transcript_id": transcript_id, **x} for x, q in zip(json_out, questions)]
    print(f"[{transcript_id}] {json_out}")
    store_result(json_out)