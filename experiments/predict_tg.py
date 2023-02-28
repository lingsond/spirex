#!/opt/conda/bin/python3
import argparse
import json
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
# from evaluate import load
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def get_prompt01():
    info_text = "From the following Context and Clickbait, extract the spoiler.\n"
    # context_text = "Context: " + context_str_new + '\n'
    context_text = "Context: "
    question_text = "Clickbait: "
    spoiler_text = "Spoiler: "
    # input_orig = info_text + context_text + question_text_orig + spoiler_text
    return info_text, context_text, question_text, spoiler_text


def get_prompt02():
    # "Answer the following question based on the given context:\n
    # context: <<context>>\n
    # question: <<question>>\n
    # answer: "
    info_text = "Answer the following question based on the given context:\n"
    context_text = "context: "
    question_text = "question: "
    spoiler_text = "answer: "
    return info_text, context_text, question_text, spoiler_text


def predict(dataset, questions, contexts, version):
    predictions = []
    for i, item in enumerate(tqdm(dataset)):
        uuid = item['uuid']
        tags = item['tags'][0]

        context_str = contexts[i]['pred']
        question_str = questions[i]['pred']

        if version == '1':
            info_text, context_text, question_text, spoiler_text = get_prompt01()
        else:
            info_text, context_text, question_text, spoiler_text = get_prompt02()

        new_context_text = context_text + context_str + '\n'
        new_question_text = question_text + question_str + '\n'
        input_text = info_text + new_question_text + new_context_text + spoiler_text
        input_ids = TOKENIZER(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.cuda()
        outputs = MODEL.generate(input_ids, max_new_tokens=50)
        output_text = TOKENIZER.decode(outputs[0])
        #     # output_text = output_text.removeprefix('<pad> ')
        #     # answer = output_text.removesuffix('</s>')
        answer = output_text[6:-4]
        infer = {'uuid': uuid, 'spoiler': answer, 'spoilerType': tags}
        predictions.append(infer)
    return predictions


def run_inference(mdl, qver, pver):
    val_file = '../../data/raw/validation.jsonl'
    flist = get_json_list(val_file)
    qfile = f'question_{qver}.jsonl'
    cfile = 'context_sorted.jsonl'
    qlist = get_json_list(qfile)
    clist = get_json_list(cfile)
    predictions = predict(flist, qlist, clist, pver)
    if mdl == 'google/flan-t5-large':
        output_file = f'./results/tg_flant5_q{qver}_p{pver}.jsonl'
    else:
        output_file = f'./results/tg_unifiedqa_q{qver}_p{pver}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as fh:
        for prediction in predictions:
            fh.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    MODELS = ['google/flan-t5-large', 'marksverdhei/unifiedqa-large-reddit-syac']

    CACHE_DIR = "../../cache"

    for model in MODELS:
        if model == 'google/flan-t5-large':
            TOKENIZER = T5Tokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
            MODEL = T5ForConditionalGeneration.from_pretrained(model, cache_dir=CACHE_DIR)
        else:
            TOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
            MODEL = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=CACHE_DIR)
        MODEL = MODEL.cuda()
        for question_version in ['0', '1', '2']:
            for prompt_version in ['1', '2']:
                run_inference(model, question_version, prompt_version)
