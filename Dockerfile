FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN pip3 install docker transformers tqdm evaluate sentencepiece bert-score

COPY run_hybrid_deberta.py run_hybrid_roberta.py download_models.py input.jsonl /
RUN python3 /download_models.py

RUN chmod +x /run_hybrid_deberta.py
ENTRYPOINT [ "/run_hybrid_deberta.py" ]

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-spirex:0.0.1