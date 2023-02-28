# A short script to check whether spoiler is always in the text
# of the paraphrased context (using semantic similarity)

import json
from transformers import T5Tokenizer


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def check_spoiler_in(j_list, c_list, qtags):
    total = 0
    total_all = 0
    for i, item in enumerate(j_list):
        spoiler = item['spoiler'][0]
        tag = item['tags'][0]
        if qtags is None:
            qtags = tag
        if tag == qtags:
            total_all += 1
            context = c_list[i]['pred']
            idx = context.find(spoiler)
            if idx >= 0:
                total += 1
    return total, total_all


def check_context(jfile, mdl, qtag):
    cfile = f'sim_context_{mdl}.jsonl'
    jlist = get_json_list(jfile)
    clist = get_json_list(cfile)

    print(f'Statistics for QA Model = {mdl} - {qtag} only')
    total, total_all = check_spoiler_in(jlist, clist, qtag)
    percentage = total / total_all
    not_found = total_all - total
    print(f'Total all: {total_all} - Total found: {total} - Total not found: {not_found} - Percentage: {percentage:.2f}')
    print()


if __name__ == '__main__':
    target_path = '../../data/raw/'
    train_file = 'train.jsonl'
    valid_file = 'validation.jsonl'
    output1 = target_path + train_file
    output2 = target_path + valid_file

    check_tags = True

    models = ['deberta', 'roberta']
    tags = ['phrase', 'passage', 'multi']
    if not check_tags:
        tags = [None]
    for tag in tags:
        for model in models:
            check_context(output2, model, tag)
