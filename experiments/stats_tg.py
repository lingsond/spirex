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


def check_spoiler(dataset, questions, contexts, version):
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

        if version == '1':
            info_text, context_text, question_text, spoiler_text = get_prompt01()
        else:
            info_text, context_text, question_text, spoiler_text = get_prompt02()

        new_context_text = context_text + context_str + '\n'
        new_question_text = question_text + question_str + '\n'
        input_text = info_text + new_question_text + new_context_text + spoiler_text
        input_ids = TOKENIZER(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        truncated_text = TOKENIZER.decode(input_ids[0])
        if version == '1':
            index = truncated_text.find('Context: ')
        else:
            index = truncated_text.find('context: ')
        truncated_context = truncated_text[index + 9:]

        idx = truncated_context.find(spoiler)
        if idx >= 0:
            total_all += 1
            if tag == 'phrase':
                total_phrase += 1
            elif tag == 'passage':
                total_passage += 1
            else:
                total_multi += 1

    return total_all, total_phrase, total_passage, total_multi


def check_context(mdl, qver, pver, cmode):
    val_file = '../../data/raw/validation.jsonl'
    flist = get_json_list(val_file)
    qfile = f'question_{qver}.jsonl'
    if cmode == 'sorted':
        cfile = 'context_sorted.jsonl'
    else:
        cfile = 'context_original.jsonl'
    qlist = get_json_list(qfile)
    clist = get_json_list(cfile)
    total, phrase, passage, multi = check_spoiler(flist, qlist, clist, pver)
    total_all = 800
    total_phrase = 335
    total_passage = 322
    total_multi = 143
    pall = total / total_all
    pphrase = phrase / total_phrase
    ppassage = passage / total_passage
    pmulti = multi / total_multi
    text = f'{mdl}-q{qver}-p{pver}\t{total}\t{pall}\t{phrase}\t{pphrase}\t{passage}\t{ppassage}\t{multi}\t{pmulti}\n'

    return text


if __name__ == '__main__':
    MODELS = ['google/flan-t5-large', 'marksverdhei/unifiedqa-large-reddit-syac']
    CACHE_DIR = "../../cache"

    model1 = 'google/flan-t5-large'
    TOKENIZER1 = T5Tokenizer.from_pretrained(model1, cache_dir=CACHE_DIR)
    MODEL1 = T5ForConditionalGeneration.from_pretrained(model1, cache_dir=CACHE_DIR)
    model2 = 'marksverdhei/unifiedqa-large-reddit-syac'
    TOKENIZER2 = AutoTokenizer.from_pretrained(model2, cache_dir=CACHE_DIR)
    MODEL2 = AutoModelForSeq2SeqLM.from_pretrained(model2, cache_dir=CACHE_DIR)

    all_stats = []
    all_mode = ['sorted']
    for mode in all_mode:
        for model in MODELS:
            if model == 'google/flan-t5-large':
                TOKENIZER = TOKENIZER1
                MODEL = MODEL1
            else:
                TOKENIZER = TOKENIZER2
                MODEL = MODEL2
            # MODEL = MODEL.cuda()
            for question_version in ['0', '1', '2']:
                for prompt_version in ['1', '2']:
                    stats = check_context(model, question_version, prompt_version, mode)
                    all_stats.append(stats)
        if mode == 'original':
            output_file = f'./stats_context_tg_baseline.tsv'
        else:
            output_file = f'./stats_context_tg.tsv'

        with open(output_file, 'w', encoding='utf-8') as fh:
            for item in all_stats:
                fh.writelines(item)
