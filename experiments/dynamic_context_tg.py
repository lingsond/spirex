import json
from transformers import T5Tokenizer, AutoTokenizer
from tqdm import tqdm


def get_json_list(json_file):
    with open(json_file, "r", encoding='utf-8') as jf:
        json_string = list(jf)
    json_list = [json.loads(x) for x in json_string]
    return json_list


def get_prompt01():
    info_text = "From the following postText and Context, extract the spoiler.\n"
    # context_text = "Context: " + context_str_new + '\n'
    context_text = "Context: "
    question_text = "postText: "
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


def get_dynamic_context(inputs, pver, qver):
    predictions = []
    items = [x['pred'] for x in inputs]
    question_file = f'question_{qver}.jsonl'
    qlist = get_json_list(question_file)
    if pver == '1':
        info_text, context_text, question_text, spoiler_text = get_prompt01()
    else:
        info_text, context_text, question_text, spoiler_text = get_prompt02()
    for x, item in enumerate(items):
        i = 1
        new_context = ''
        input_text = ''
        context_len = 0
        question = qlist[x]['pred']
        new_question_text = question_text + question + '\n'
        while context_len < 512 and i < len(item):
            input_text = new_context
            new_context = input_text + ' ' + item[i]
            new_context_text = context_text + new_context + '\n'
            new_text = info_text + new_context_text + new_question_text + spoiler_text
            input_ids = TOKENIZER(new_text, return_tensors="pt").input_ids
            context_len = list(input_ids.size())[1]
            i += 1

        infer = {'pred': input_text}
        predictions.append(infer)
    return predictions


def get_sorted_context(inputs, pver, qver):
    predictions = []
    items = [x['pred'] for x in inputs]
    question_file = f'question_{qver}.jsonl'
    qlist = get_json_list(question_file)
    if pver == '1':
        info_text, context_text, question_text, spoiler_text = get_prompt01()
    else:
        info_text, context_text, question_text, spoiler_text = get_prompt02()
    for x, item in enumerate(items):
        context = ' '.join(item)
        question = qlist[x]['pred']
        new_question_text = question_text + question + '\n'
        new_context_text = context_text + context + '\n'
        new_prompt = info_text + new_question_text + new_context_text + spoiler_text

        infer = {'pred': new_prompt}
        predictions.append(infer)
    return predictions


def run_inference(model_name, prompt_ver, question_ver):
    input_file = 'sim_context_all.jsonl'
    inp = [json.loads(i) for i in open(input_file, 'r', encoding='utf-8')]
    contexts = get_sorted_context(inp, prompt_ver, question_ver)
    if model_name == 'google/flan-t5-large':
        output_file = f'sim_context_flant5_q{question_ver}_p{prompt_ver}.jsonl'
    else:
        output_file = f'sim_context_unifiedqa_q{question_ver}_p{prompt_ver}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as fh:
        for item in contexts:
            fh.write(json.dumps(item) + '\n')


def check_context_length(model_name, prompt_ver, question_ver):
    if model_name == 'google/flan-t5-large':
        input_file = f'sim_context_flant5_q{question_ver}_p{prompt_ver}.jsonl'
    else:
        input_file = f'sim_context_unifiedqa_q{question_ver}_p{prompt_ver}.jsonl'
    inputs = [json.loads(i) for i in open(input_file, 'r', encoding='utf-8')]
    items = [x['pred'] for x in inputs]
    for i, item in enumerate(tqdm(items)):
        input_ids = TOKENIZER(item, return_tensors="pt").input_ids
        context_len = list(input_ids.size())[1]
        if context_len > 512:
            print(f'{model_name} - q{question_ver}-p{prompt_ver} - i: {i}')


if __name__ == '__main__':
    MODELS = ['google/flan-t5-large', 'marksverdhei/unifiedqa-large-reddit-syac']

    CACHE_DIR = "../../cache"

    for model in MODELS:
        if model == 'google/flan-t5-large':
            TOKENIZER = T5Tokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        else:
            TOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir=CACHE_DIR)
        for question_version in ['0', '1', '2']:
            for prompt_version in ['1', '2']:
                run_inference(model, prompt_version, question_version)
                check_context_length(model, prompt_version, question_version)
