import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def save_original_context(jfile):
    jlist = get_json_list(jfile)
    prediction = []
    for i, item in enumerate(tqdm(jlist)):
        query = item['postText'][0]
        corpus = item['targetParagraphs']
        title = item['targetTitle']
        if query != title and title != corpus[0]:
            new_corpus = [title]
            new_corpus.extend(corpus)
        else:
            new_corpus = corpus

        corpus_text = ' '.join(new_corpus)
        infer = {'pred': corpus_text}
        prediction.append(infer)

    output = f'context_original.jsonl'
    with open(output, 'w', encoding='utf-8') as fh:
        for item in prediction:
            fh.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    target_path = '../../data/raw/'
    train_file = 'train.jsonl'
    valid_file = 'validation.jsonl'
    output1 = target_path + train_file
    output2 = target_path + valid_file

    save_original_context(output2)
