import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def para_question(question_text: str, version: int):
    if question_text.endswith('?'):
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


def get_question(inputs, ver):
    predictions = []
    for item in tqdm(inputs):
        question_orig = item['postText'][0]
        if ver > 0:
            question_para = para_question(question_orig, ver)
        else:
            question_para = question_orig

        infer = {'pred': question_para}
        predictions.append(infer)
    return predictions


def run_inference(input_file):
    inp = [json.loads(i) for i in open(input_file, 'r', encoding='utf-8')]
    versions = [0, 1, 2]
    for version in versions:
        questions = get_question(inp, version)
        output_file = f'question_{str(version)}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as fh:
            for item in questions:
                fh.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    target_path = '../../data/raw/'
    train_file = 'train.jsonl'
    valid_file = 'validation.jsonl'
    output1 = target_path + train_file
    output2 = target_path + valid_file

    CACHE_DIR = "../../cache"

    # Model Initialization
    PROMPT_MODEL = 'google/flan-t5-large'
    PTOKENIZER = T5Tokenizer.from_pretrained(PROMPT_MODEL, cache_dir=CACHE_DIR)
    PMODEL = T5ForConditionalGeneration.from_pretrained(PROMPT_MODEL, cache_dir=CACHE_DIR)
    SIM_MODEL = 'all-MiniLM-L6-v2'
    SIM_EMBEDDER = SentenceTransformer(SIM_MODEL, cache_folder=CACHE_DIR)

    run_inference(output2)
