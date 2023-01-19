from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, \
    AutoTokenizer, AutoModelForQuestionAnswering
from evaluate import load


if __name__ == '__main__':
    # Model Initialization
    PROMPT_MODEL = ['google/flan-t5-large']
    for model in PROMPT_MODEL:
        PTOKENIZER = T5Tokenizer.from_pretrained(model, cache_dir="/.cache")
        PMODEL = T5ForConditionalGeneration.from_pretrained(model, cache_dir="/.cache")
    QA_MODEL = ['deepset/deberta-v3-large-squad2', 'deepset/roberta-large-squad2']
    for model in QA_MODEL:
        QTOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir="/.cache")
        qmodel = AutoModelForQuestionAnswering.from_pretrained(model, cache_dir="/.cache")
    # QMODEL = pipeline("question-answering", model=qmodel, tokenizer=QTOKENIZER)
    BERTSCORE = load("bertscore", cache_dir="/.cache")
