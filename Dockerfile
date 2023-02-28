FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV TRANSFORMER_CACHE=/spirex/cache/
# RUN pip3 install docker transformers tqdm evaluate sentencepiece bert-score
RUN pip3 install docker transformers tqdm sentencepiece sentence-transformers

WORKDIR /spirex/
COPY download_models.py /spirex/
RUN python3 /spirex/download_models.py

COPY qa_deberta0.py qa_deberta1.py qa_deberta2.py input.jsonl /spirex/
COPY qa_deberta0_sorted.py qa_deberta1_sorted.py qa_deberta2_sorted.py /spirex/
COPY tg01.py tg01_sorted.py tg11.py tg11_sorted.py tg12.py tg12_sorted.py /spirex/
COPY hybrid_q1tg12.py hybrid_q1tg12_sorted.py /spirex/
COPY hybrid_q1tg11.py hybrid_q1tg11_sorted.py /spirex/

RUN chmod -R 777 /spirex/

ENTRYPOINT [ "/spirex/qa_deberta0.py" ]

# NAME=ls6...-spirex:0.0.3
# webis...:spirex:0.0.6