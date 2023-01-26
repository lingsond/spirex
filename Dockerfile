FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV TRANSFORMER_CACHE=/spirex/cache/
# RUN pip3 install docker transformers tqdm evaluate sentencepiece bert-score
RUN pip3 install docker transformers tqdm sentencepiece sentence-transformers

WORKDIR /spirex/
COPY run_hybrid_deberta.py run_hybrid_roberta.py run_prompt.py input.jsonl /spirex/
COPY download_models.py /spirex/
RUN python3 /spirex/download_models.py

COPY new_hybrid_deberta.py new_hybrid_roberta.py /spirex/
COPY run_qa_no_para_deberta.py run_qa_no_para_roberta.py /spirex/
COPY run_prompt_only_no_para.py /spirex/
COPY run_qa_para_question_deberta.py run_qa_para_question_roberta.py /spirex/
COPY run_prompt_only_para_question.py /spirex/
# RUN chmod +x /spirex/new_hybrid_deberta.py
# RUN chmod +x /spirex/new_hybrid_roberta.py
# RUN chmod +x /spirex/run_qa_no_para_deberta.py
# RUN chmod +x /spirex/run_qa_no_para_roberta.py
# RUN chmod +x /spirex/run_prompt_only_no_para.py
# RUN chmod +x /spirex/run_prompt.py
RUN chmod -R 777 /spirex/
# RUN mkdir /.cache
# RUN chmod -R 777 /.cache/

ENTRYPOINT [ "/spirex/run_hybrid_deberta.py" ]

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-spirex:0.0.2
# webis...:spirex:0.0.3