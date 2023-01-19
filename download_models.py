from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import load


if __name__ == '__main__':
    # Model Initialization
    PROMPT_MODEL = 'google/flan-t5-large'
    QA_MODEL = 'deepset/deberta-v3-large-squad2'
    PTOKENIZER = T5Tokenizer.from_pretrained(PROMPT_MODEL, cache_dir="./cache")
    PMODEL = T5ForConditionalGeneration.from_pretrained(PROMPT_MODEL, cache_dir="./cache")
    QTOKENIZER = AutoTokenizer.from_pretrained(QA_MODEL, cache_dir="./cache")
    qmodel = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL, cache_dir="./cache")
    QMODEL = pipeline("question-answering", model=qmodel, tokenizer=QTOKENIZER)
    BERTSCORE = load("bertscore", cache_dir="./cache")
