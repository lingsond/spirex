import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def save_similar_context(jfile):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    jlist = get_json_list(jfile)
    prediction = []
    for i, item in enumerate(tqdm(jlist)):
        query = item['postText'][0]
        corpus = item['targetParagraphs']
        title = item['targetTitle']
        if query != title and title != corpus[0]:
            corpus.append(title)
        corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        top_k = len(corpus)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]
        new_corpus = []
        for hit in hits:
            new_corpus.append(corpus[hit['corpus_id']])
        infer = {'pred': new_corpus}
        prediction.append(infer)

    output = f'sim_context_all.jsonl'
    with open(output, 'w', encoding='utf-8') as fh:
        for item in prediction:
            fh.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    target_path = '../../data/raw/'
    train_file = 'train.jsonl'
    valid_file = 'validation.jsonl'
    output1 = target_path + train_file
    output2 = target_path + valid_file

    save_similar_context(output2)
