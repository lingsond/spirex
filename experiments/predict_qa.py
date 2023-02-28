#!/opt/conda/bin/python3
import argparse
import json
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering
# from evaluate import load
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def get_qa(question_text, context):
    ans = MODEL(
        question=question_text, context=context,
        max_answer_len=30, max_seq_len=512
    )
    return ans['answer']


def predict(dataset, questions, contexts):
    predictions = []
    for i, item in enumerate(tqdm(dataset)):
        uuid = item['uuid']
        tags = item['tags'][0]

        context_str = contexts[i]['pred']
        if len(context_str) == 0:
            context_str = '...'
        question_str = questions[i]['pred']
        answer = get_qa(question_str, context_str)

        infer = {'uuid': uuid, 'spoiler': answer, 'spoilerType': tags}
        predictions.append(infer)
    return predictions


def run_inference(mdl, qver):
    val_file = '../../data/raw/validation.jsonl'
    flist = get_json_list(val_file)
    qfile = f'question_{qver}.jsonl'
    if mdl == 'deepset/deberta-v3-large-squad2':
        cfile = 'sim_context_deberta1.jsonl'
    else:
        cfile = 'sim_context_roberta1.jsonl'
    qlist = get_json_list(qfile)
    clist = get_json_list(cfile)
    predictions = predict(flist, qlist, clist)
    if mdl == 'deepset/deberta-v3-large-squad2':
        output_file = f'./results/qa_deberta1_q{qver}.jsonl'
    else:
        output_file = f'./results/qa_roberta1_q{qver}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as fh:
        for prediction in predictions:
            fh.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    MODELS = ['deepset/deberta-v3-large-squad2', 'deepset/roberta-large-squad2']

    CACHE_DIR = "../../cache"

    for model in MODELS:
        TOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        qmodel = AutoModelForQuestionAnswering.from_pretrained(model, cache_dir=CACHE_DIR)
        MODEL = pipeline("question-answering", model=qmodel, tokenizer=TOKENIZER, device=0)
        for question_version in ['0', '1', '2']:
            run_inference(model, question_version)
