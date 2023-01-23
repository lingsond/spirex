FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV TRANSFORMER_CACHE=/spirex/cache/
# RUN pip3 install docker transformers tqdm evaluate sentencepiece bert-score
RUN pip3 install docker transformers tqdm sentencepiece sentence-transformers

WORKDIR /spirex/
COPY run_hybrid_deberta.py run_hybrid_roberta.py run_prompt.py input.jsonl /spirex/
COPY download_models.py /spirex/
RUN python3 /spirex/download_models.py

RUN chmod +x /spirex/run_hybrid_deberta.py
RUN chmod +x /spirex/run_hybrid_roberta.py
RUN chmod -R 777 /spirex/cache/
# RUN mkdir /.cache
RUN chmod -R 777 /.cache/

ENTRYPOINT [ "/spirex/run_hybrid_deberta.py" ]

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-spirex:0.0.2