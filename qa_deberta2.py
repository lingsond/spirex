#!/opt/conda/bin/python3
import argparse
import json
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering
# from evaluate import load
from sentence_transformers import SentenceTransformer, util


def parse_args():
    parser = argparse.ArgumentParser(description='This is a model for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def para_question(question_text: str, version: int):
    if question_text.endswith('?') or version == 0:
        pred = question_text
    else:
        if version == 1:
            new_text = 'Change the headline to question form. Headline: ' + question_text
        else:
            new_text = f'Rephrase as a question: {question_text}\nquestion: '
        input_ids = PTOKENIZER(new_text, return_tensors="pt").input_ids
        outputs = PMODEL.generate(input_ids, max_new_tokens=50)
        question = PTOKENIZER.decode(outputs[0])
        # question = question.removeprefix('<pad> ')
        question = question[6:-4]
        #question = question.removesuffix('</s>')

        # Check Sim Score
        orig_embedding = SIM_EMBEDDER.encode(question_text)
        para_embedding = SIM_EMBEDDER.encode(question)
        hits = util.semantic_search(para_embedding, orig_embedding, top_k=1)
        hit = hits[0][0]
        f1 = hit['score']

        if f1 > 0.7:
            pred = question
        else:
            pred = question_text
    return pred


def get_qa(question_text, content, ptag):
    tag_text = "\nAnswer type is a " + ptag
    # context_text = content + tag_text
    context_text = content
    ans = QMODEL(
        question=question_text, context=context_text,
        max_answer_len=30, max_seq_len=512
    )
    return ans['answer']


def predict(inputs, vers):
    predictions = []
    for item in inputs:
        uuid = item['uuid']
        question_orig = item['postText'][0]
        paragraphs = item['targetParagraphs']
        title = item['targetTitle']
        tags = item['tags'][0]
        context_list = [title]
        if title != question_orig and title != paragraphs[0]:
            context_list.extend(paragraphs)
        else:
            context_list = paragraphs
        context_str = '\n'.join(context_list)
        question_para = para_question(question_orig, vers)

        answer = get_qa(question_para, context_str, tags)

        infer = {'uuid': uuid, 'spoiler': answer, 'spoilerType': tags}
        predictions.append(infer)
    return predictions


def run_inference(input_file, output_file, version):
    inp = [json.loads(i) for i in open(input_file, 'r')]
    predictions = predict(inp, version)
    with open(output_file, 'w', encoding='utf-8') as fh:
        for prediction in predictions:
            fh.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()

    CACHE_DIR = "/spirex/cache/"
    # CACHE_DIR = "./cache"

    # Model Initialization
    PROMPT_MODEL = 'google/flan-t5-large'
    QA_MODEL = 'deepset/deberta-v3-large-squad2'
    PTOKENIZER = T5Tokenizer.from_pretrained(PROMPT_MODEL, cache_dir=CACHE_DIR)
    PMODEL = T5ForConditionalGeneration.from_pretrained(PROMPT_MODEL, cache_dir=CACHE_DIR)
    QTOKENIZER = AutoTokenizer.from_pretrained(QA_MODEL, cache_dir=CACHE_DIR)
    qmodel = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL, cache_dir=CACHE_DIR)
    QMODEL = pipeline("question-answering", model=qmodel, tokenizer=QTOKENIZER)
    SIM_MODEL = 'all-MiniLM-L6-v2'
    SIM_EMBEDDER = SentenceTransformer(SIM_MODEL, cache_folder=CACHE_DIR)
    # BERTSCORE = load("bertscore", cache_dir="/spirex/cache/")

    ver = 2
    run_inference(args.input, args.output, ver)
