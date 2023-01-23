from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForMaskedLM
)
# from evaluate import load
from sentence_transformers import SentenceTransformer, util


if __name__ == '__main__':
    # Model Initialization
    PROMPT_MODEL = ['google/flan-t5-large']
    for model in PROMPT_MODEL:
        PTOKENIZER = T5Tokenizer.from_pretrained(model, cache_dir="/spirex/cache/")
        PMODEL = T5ForConditionalGeneration.from_pretrained(model, cache_dir="/spirex/cache/")
    QA_MODEL = ['deepset/deberta-v3-large-squad2', 'deepset/roberta-large-squad2']
    for model in QA_MODEL:
        QTOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir="/spirex/cache")
        qmodel = AutoModelForQuestionAnswering.from_pretrained(model, cache_dir="/spirex/cache")
    #SCORE_MODEL = ['distilbert-base-uncased']
    #for model in SCORE_MODEL:
    #    STOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir="/.cache/huggingface/hub/")
    #    smodel = AutoModelForMaskedLM.from_pretrained(model, cache_dir="/spirex/cache")
    # QMODEL = pipeline("question-answering", model=qmodel, tokenizer=QTOKENIZER)
    # BERTSCORE = load("bertscore", cache_dir="/spirex/cache")
    #bscore = BERTSCORE.compute(
    #    predictions=["Annakin Skywalker"], references=["Darth Vader"],
    #    model_type="distilbert-base-uncased"
    #)
    embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="/spirex/cache")
