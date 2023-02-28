import json
from transformers import AutoTokenizer
from tqdm import tqdm


def get_dynamic_context(inputs):
    predictions = []
    items = [x['pred'] for x in inputs]
    for x, item in enumerate(items):
        context = ' '.join(item)
        infer = {'pred': context}
        predictions.append(infer)
    return predictions


def run_inference(model_name):
    input_file = 'sim_context_all.jsonl'
    inp = [json.loads(i) for i in open(input_file, 'r', encoding='utf-8')]
    contexts = get_dynamic_context(inp)
    if model_name == 'deepset/deberta-v3-large-squad2':
        output_file = 'sim_context_deberta1.jsonl'
    else:
        output_file = 'sim_context_roberta1.jsonl'
    with open(output_file, 'w', encoding='utf-8') as fh:
        for item in contexts:
            fh.write(json.dumps(item) + '\n')


def check_context_length(model_name):
    if model_name == 'deepset/deberta-v3-large-squad2':
        input_file = 'sim_context_deberta.jsonl'
    else:
        input_file = 'sim_context_roberta.jsonl'
    inputs = [json.loads(i) for i in open(input_file, 'r')]
    items = [x['pred'] for x in inputs]
    for i, item in enumerate(tqdm(items)):
        input_ids = TOKENIZER(item, return_tensors="pt").input_ids
        context_len = list(input_ids.size())[1]
        if context_len > 512:
            print(f'{model_name} - i: {i}')


if __name__ == '__main__':
    MODELS = ['deepset/deberta-v3-large-squad2', 'deepset/roberta-large-squad2']

    CACHE_DIR = "../../cache"

    for model in MODELS:
        TOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        run_inference(model)
        # check_context_length(model)
