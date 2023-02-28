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
        input_text = info_text + new_context_text + new_question_text + spoiler_text
        input_ids = TOKENIZER(input_text, return_tensors="pt").input_ids.cuda()
        outputs = MODEL.generate(input_ids, max_new_tokens=50)
        output_text = TOKENIZER.decode(outputs[0])
        #     # output_text = output_text.removeprefix('<pad> ')
        #     # answer = output_text.removesuffix('</s>')
        answer = output_text[6:-4]
        infer = {'uuid': uuid, 'spoiler': answer, 'spoilerType': tags}
        predictions.append(infer)
    return predictions


def get_result(mdl, qver):
    input_file = f'output_{mdl}1_q{qver}.txt'
    # input_file = f'baseline_{mdl}_q{qver}.txt'
    with open(input_file, 'r') as fh:
        text = fh.readlines()
    values_raw = [text[6], text[10], text[22], text[26], text[38], text[42], text[54], text[58]]
    values_str = [x[10:-2] for x in values_raw]
    values = [float(x) for x in values_str]
    result = f'{input_file}\t{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}\t' \
             f'{values[3]:.4f}\t{values[4]:.4f}\t{values[5]:.4f}\t{values[6]:.4f}\t{values[7]:.4f}'
    return result


if __name__ == '__main__':
    MODELS = ['deberta', 'roberta']
    qvers = ['0', '1', '2']

    results = []
    for model in MODELS:
        for qver in qvers:
            result = get_result(model, qver)
            results.append(result)
    with open('output1_qa.tsv', 'w') as fh:
    #with open('baseline_qa.tsv', 'w') as fh:
        for item in results:
            fh.writelines(item + '\n')
