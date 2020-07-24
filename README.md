# GoogleNQ_Challenge

Installations required-
1. torch
2. transformers (HuggingFace)
3. jsonlines
4. absl-py

Steps to run-
0. Clone the repo.
1. Download simplified train data 'v1.0-simplified_simplified-nq-train.jsonl.gz' and place in data/train.
2. Run 'python small_data.py' to get 'simplified-nq-train.jsonl' in data/train.
3. Open train.ipynb and run all cells.
4. After this we will have 4 saved models in 'models/'.
5. Rename 'model_100k3.pth' to 'pytorch_model.bin'.
6. Open test.ipynb and run all cells. The prediction json file will be created.
7. For evaluation, run 'python nq_eval.py --gold_path=data/dev/v1.0_sample_nq-dev-sample.jsonl.gz --predictions_path=predictions.json'.

Score-
Current model, after training for 4 epochs (on 100k questions) gives F1 about 0.37. This evaluation has been 
done on the sample dev dataset, not the complete dev dataset.
 
Architecture and design choices-
For this task, owing to transfer learning I started with a BERT large model pretrained on the SQuAD set. 
Initial model involved giving the BERT an input of [CLS]"question"[SEP]"document_text"[SEP],
and extracting start and end tokens within the context (document text) that answered the question.
However, my intuition was that with the large sizes of document text, BERT wouldnt be able
to generate very useful embeddings.

Next, I took some ideas from the Kaggle challenge for Google NQ dataset. 
Here, as the input I give [CLS]"question"[SEP]"long answer candidate"[SEP]. 
The dataset & dataloader collate the input in such a way, that for each question a positive instance
is created i.e. [CLS]"question"[SEP]"positive long answer candidate"[SEP] where positive candidate is 
a long answer which is also found in the annotations.
For each question, a negative instance is also created i.e.
 [CLS]"question"[SEP]"negative long answer candidate"[SEP] where negative candidate is one which does not
belong to the annotations.
So for N questions, we train over 2N instances.
The span prediction here predicts short answer start and end tokens within these long answer candidates.
After this, there is a classifier layer which indicates whether the long answer candidate in the current
instance is a valid answer or not by 0 for NO ANSWER and 1 for LONG ANSWER.
Then taking the cross entropy loss of short answer start logits, end logits and class logits with gold labels,
we propagate the sum of these losses backward and optimize. 
The reason for including this was to benefit from multitask learning where learning 2 similar tasks
can benefit their performances.

For evaluation,
I relied on the output of the final classifier layer. Here I evaluated over each candidate of each question.
The candidate with the highest long answer score was selected and its long answer start and end tokens were
stored as the prediction.

The results I received were below my expectations of the model, which I believe is due to the simplistic
evaluation procedure. Also, with more compute power, I would train it over the entire dataset distributedly.
Also, try to use ensemble models for better inference.

Feedback about the challenge-
It was challenging and enjoyable. Would have enjoyed spending some more time on it and getting better results.

Charity-
Pennsylvania Women Work (https://greatnonprofits.org/org/pennsylvania-women-work)
