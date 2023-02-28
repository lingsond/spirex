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


def check_spoiler(dataset, questions, contexts):
    total_all = 0
    total_phrase = 0
    total_passage = 0
    total_multi = 0
    for i, item in enumerate(tqdm(dataset)):
        uuid = item['uuid']
        tag = item['tags'][0]
        spoiler = item['spoiler'][0]

        context_str = contexts[i]['pred']
        question_str = questions[i]['pred']

        input_ids = TOKENIZER(context_str, return_tensors="pt", max_length=512, truncation=True).input_ids
        truncated_text = TOKENIZER.decode(input_ids[0])

        idx = truncated_text.find(spoiler)
        if idx >= 0:
            total_all += 1
            if tag == 'phrase':
                total_phrase += 1
            elif tag == 'passage':
                total_passage += 1
            else:
                total_multi += 1

    return total_all, total_phrase, total_passage, total_multi


def check_context(mdl, qver, cmode):
    val_file = '../../data/raw/validation.jsonl'
    flist = get_json_list(val_file)
    qfile = f'question_{qver}.jsonl'
    if cmode == 'sorted':
        cfile = 'context_sorted.jsonl'
    else:
        cfile = 'context_original.jsonl'
    qlist = get_json_list(qfile)
    clist = get_json_list(cfile)
    total, phrase, passage, multi = check_spoiler(flist, qlist, clist)
    total_all = 800
    total_phrase = 335
    total_passage = 322
    total_multi = 143
    pall = total / total_all
    pphrase = phrase / total_phrase
    ppassage = passage / total_passage
    pmulti = multi / total_multi
    text = f'{mdl}-q{qver}\t{total}\t{pall}\t{phrase}\t{pphrase}\t{passage}\t{ppassage}\t{multi}\t{pmulti}\n'

    return text


if __name__ == '__main__':
    MODELS = ['deepset/deberta-v3-large-squad2', 'deepset/roberta-large-squad2']
    CACHE_DIR = "../../cache"

    model1 = 'deepset/deberta-v3-large-squad2'
    TOKENIZER1 = AutoTokenizer.from_pretrained(model1, cache_dir=CACHE_DIR)
    #qmodel1 = AutoModelForQuestionAnswering.from_pretrained(model1, cache_dir=CACHE_DIR)
    #MODEL1 = pipeline("question-answering", model=qmodel1, tokenizer=TOKENIZER1, device=0)
    model2 = 'deepset/roberta-large-squad2'
    TOKENIZER2 = AutoTokenizer.from_pretrained(model2, cache_dir=CACHE_DIR)
    #qmodel2 = AutoModelForQuestionAnswering.from_pretrained(model2, cache_dir=CACHE_DIR)
    #MODEL2 = pipeline("question-answering", model=qmodel2, tokenizer=TOKENIZER2, device=0)

    all_mode = ['original', 'sorted']
    for mode in all_mode:
        all_stats = []
        for model in MODELS:
            if model == 'deepset/deberta-v3-large-squad2':
                TOKENIZER = TOKENIZER1
                #MODEL = MODEL1
            else:
                TOKENIZER = TOKENIZER2
                #MODEL = MODEL2
            # MODEL = MODEL.cuda()
            for question_version in ['0', '1', '2']:
                stats = check_context(model, question_version, mode)
                all_stats.append(stats)
        if mode == 'original':
            output_file = f'./stats_context_qa_baseline.tsv'
        else:
            output_file = f'./stats_context_qa.tsv'

        with open(output_file, 'w', encoding='utf-8') as fh:
            for item in all_stats:
                fh.writelines(item)
