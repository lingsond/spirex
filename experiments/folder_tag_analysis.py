import json
# import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
# from rouge import rouge_n_sentence_level, rouge_l_sentence_level
from rouge import Rouge
from evaluate import load
from pathlib import Path


def get_files_in_folder(folder_path: Path):
    file_list = []
    for item in folder_path.iterdir():
        if item.is_file():
            file_list.append(item)

    return file_list


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_list = list(jf)
    return json_list


def get_exact_match_score(results, base):
    analysis = []
    score1 = 0
    score2 = 0
    for i, result in enumerate(results):
        # result = json.loads(json_str)
        answer1 = result['spoiler']
        answer2 = answer1.strip()
        ans1_token = word_tokenize(answer1)
        ans2_token = word_tokenize(answer2)
        gold = base[i]['spoiler']
        gold_text = ' '.join(gold)
        gold_token = word_tokenize(gold_text)
        if ans1_token == gold_token:
            score1 += 1
        if ans2_token == gold_token:
            score2 += 1
    return score1, score2


def get_rouge_score(results, base):
    ROUGE = Rouge()
    score1 = []
    score2 = []
    for i, result in enumerate(results):
        # result = json.loads(json_str)
        answer1 = result['spoiler']
        if len(set(answer1)) == 1 and answer1[0] == '.':
            answer1 = answer1.replace('.', ';')
        if not answer1:
            answer1 = ';'
        # answer2 = answer1.strip()
        gold = base[i]['spoiler']
        gold_text = ' '.join(gold)
        rscore = ROUGE.get_scores(answer1, gold_text, ignore_empty=True)
        score1.append(rscore)
        # score2.append(ROUGE.get_scores(answer2, gold_text))
    return score1


def calculate_total_rouge(rscore):
    rlen = len(rscore)
    p = [x['p'] for x in rscore]
    r = [x['r'] for x in rscore]
    f = [x['f'] for x in rscore]
    trp = sum(p) / rlen * 100
    trr = sum(r) / rlen * 100
    trf = sum(f) / rlen * 100
    return trp, trr, trf


def get_bertscore(results, base):
    bertscore = load("bertscore")
    pscore = 0
    rscore = 0
    fscore = 0
    preds = []
    golds = []
    for i, result in enumerate(results):
        # result = json.loads(json_str)
        answer = result['spoiler']
        gold = base[i]['spoiler']
        gold_text = ' '.join(gold)
        preds.append(answer)
        golds.append(gold_text)
    bscore = bertscore.compute(predictions=preds, references=golds, model_type="distilbert-base-uncased")
    precision = bscore['precision']
    recall = bscore['recall']
    f1 = bscore['f1']
    return precision, recall, f1


def get_bleu_score(results, base):
    bleu = load("bleu")
    preds = []
    golds = []
    for i, result in enumerate(results):
        # result = json.loads(json_str)
        answer = result['spoiler']
        gold = base[i]['spoiler']
        gold_text = ' '.join(gold)
        preds.append(answer)
        golds.append(gold_text)
    bscore = bleu.compute(predictions=preds, references=golds)
    return bscore['bleu']


def process_result(fpath: Path, tag_name):
    gold_file = '../../data/raw/validation.jsonl'
    gold_str = get_json_list(gold_file)
    gold_list = [json.loads(x) for x in gold_str]
    gold = [x for x in gold_list if x['tags'][0] == tag_name]
    fname = fpath.name
    fname_nosuffix = '.'.join(fname.split('.')[:-1])
    # model_name = '_'.join(fname_nosuffix.split("_")[1:])
    model_name = fname_nosuffix
    str_list = get_json_list(fpath)
    json_list = [json.loads(x) for x in str_list]
    jlist = [x for x in json_list if x['spoilerType'] == tag_name]

    exact1, exact2 = get_exact_match_score(jlist, gold)
    exact1_score = exact1 / len(jlist) * 100
    exact2_score = exact2 / len(jlist) * 100

    rg = get_rouge_score(jlist, gold)
    rg1 = [x[0]['rouge-1'] for x in rg]
    sumr1 = calculate_total_rouge(rg1)
    rg2 = [x[0]['rouge-2'] for x in rg]
    sumr2 = calculate_total_rouge(rg2)
    rgl = [x[0]['rouge-l'] for x in rg]
    sumrl = calculate_total_rouge(rgl)

    bp, br, bf = get_bertscore(jlist, gold)
    tbp = sum(bp) / len(jlist) * 100
    tbr = sum(br) / len(jlist) * 100
    tbf = sum(bf) / len(jlist) * 100

    bleu = get_bleu_score(jlist, gold)

    text = f'{model_name}\t{exact1_score:.2f}\t{exact2_score:.2f}\t' \
           f'{sumr1[0]:.2f}\t{sumr1[1]:.2f}\t{sumr1[2]:.2f}\t' \
           f'{sumr2[0]:.2f}\t{sumr2[1]:.2f}\t{sumr2[2]:.2f}\t' \
           f'{sumrl[0]:.2f}\t{sumrl[1]:.2f}\t{sumrl[2]:.2f}\t' \
           f'{tbp:.2f}\t{tbr:.2f}\t{tbf:.2f}\t{bleu:.4f}'
    return text


if __name__ == '__main__':
    # folder_name = Path('./results/')
    folder_name = Path('./crosscheck/')
    file_names = get_files_in_folder(folder_name)

    tags = ['phrase', 'passage', 'multi']
    for tag in tags:
        analysis = []
        for file_name in tqdm(file_names):
            txt = process_result(file_name, tag)
            analysis.append(txt)

        outname = f'./evaluation/result_crosscheck_old_metrics-{tag}.csv'
        with open(outname, 'w', encoding='utf-8') as fh:
            for item in analysis:
                fh.writelines(item + '\n')

