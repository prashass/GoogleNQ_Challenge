# GoogleNQ_Challenge

Installations required-
1. torch
2. transformers (HuggingFace)
3. jsonlines
4. absl-py

Steps to run-
1. Download simplified train data 'v1.0-simplified_simplified-nq-train.jsonl.gz' and place in data/train.
2. Run 'python small_data.py' to get 'simplified-nq-train.jsonl' in data/train.
3. Open train.ipynb and run all cells.
4. After this we will have 4 saved models in 'models/'.
5. Rename 'model_100k2.pth' to 'pytorch_model.bin'.
6. Open test.ipynb and run all cells. The prediction json file will be created.
7. For evaluation, run 'python nq_eval.py --gold_path=data/dev/v1.0_sample_nq-dev-sample.jsonl.gz --predictions_path=predictions.json'.

Score-
Current model, after training for 3 epochs gives F1 0.34444444444444444. Prediction file is attached. This evaluation has been done on the sample dev dataset, not the complete dev dataset.
 
Architecture and design choices-
For this task, I started with a BERT large model pretrained on the SQuAD set. 
Initial model involved giving the BERT an input of [CLS]<question>[SEP]<document_text>[SEP],
and extracting start and end tokens within 
