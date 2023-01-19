#!/usr/bin/env python3
import argparse
import json
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import load


def parse_args():
    parser = argparse.ArgumentParser(description='This is a model for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def para_question(question_text: str):
    if question_text.endswith('?'):
        pred = question_text
    else:
        new_text = 'Change the headline to question form. Headline: ' + question_text
        input_ids = PTOKENIZER(new_text, return_tensors="pt").input_ids
        outputs = PMODEL.generate(input_ids, max_new_tokens=50)
        question = PTOKENIZER.decode(outputs[0])
        question = question.removeprefix('<pad> ')
        question = question.removesuffix('</s>')

        # Check BERTscore
        bscore = BERTSCORE.compute(
            predictions=[question], references=[question_text],
            model_type="distilbert-base-uncased"
        )
        f1 = bscore['f1'][0]
        if f1 > 0.7:
            pred = question
        else:
            pred = question_text
    return pred


def get_qa(question_text, content, ptag):
    tag_text = "\nSpoiler text type is a " + ptag
    context_text = content + tag_text
    ans = QMODEL(
        question=question_text, context=context_text,
        max_answer_len=30, max_seq_len=512
    )
    return ans['answer']


def predict(inputs):
    predictions = []
    for item in inputs:
        uuid = item['uuid']
        question_orig = item['postText'][0]
        paragraphs = item['targetParagraphs']
        title = item['targetTitle']
        tags = item['tags'][0]
        context_list = [title]
        if title == paragraphs[0]:
            context_list = paragraphs
        else:
            context_list.extend(paragraphs)
        context_str = '\n'.join(context_list)
        question_para = para_question(question_orig)

        if tags == 'phrase':
            answer = get_qa(question_para, context_str, tags)
        else:
            context_text = "Context: " + context_str
            spoiler_text = "Spoiler: "
            input_text = context_text + '\n' + question_para + '\n' + spoiler_text
            if tags == 'multi':
                input_text = context_text + '\n' + question_orig + '\n' + spoiler_text
            input_ids = PTOKENIZER(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
            context_len = list(input_ids.size())[1]
            if context_len > 512:
                answer = get_qa(question_para, context_str, tags)
            else:
                outputs = PMODEL.generate(input_ids, max_new_tokens=50)
                output_text = PTOKENIZER.decode(outputs[0])
                output_text = output_text.removeprefix('<pad> ')
                answer = output_text.removesuffix('</s>')
        infer = {'uuid': uuid, 'spoiler': answer, 'spoilerType': tags}
        predictions.append(infer)
    return predictions


def run_inference(input_file, output_file):
    inp = [json.loads(i) for i in open(input_file, 'r')]
    predictions = predict(inp)
    with open(output_file, 'w', encoding='utf-8') as fh:
        for prediction in predictions:
            fh.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()

    # Model Initialization
    PROMPT_MODEL = 'google/flan-t5-large'
    QA_MODEL = 'deepset/deberta-v3-large-squad2'
    PTOKENIZER = T5Tokenizer.from_pretrained(PROMPT_MODEL, cache_dir="./cache")
    PMODEL = T5ForConditionalGeneration.from_pretrained(PROMPT_MODEL, cache_dir="./cache")
    QTOKENIZER = AutoTokenizer.from_pretrained(QA_MODEL, cache_dir="./cache")
    qmodel = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL, cache_dir="./cache")
    QMODEL = pipeline("question-answering", model=qmodel, tokenizer=QTOKENIZER)
    BERTSCORE = load("bertscore", cache_dir="./cache")

    run_inference(args.input, args.output)
