import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import jsonlines
#from test_utils import simplify_nq_example
import json
import gzip


json_dir = 'v1.0-simplified_simplified-nq-train.jsonl.gz'
dict_list = []
max_num_data = 100000
with gzip.open(json_dir) as f:
    for i, line in tqdm(enumerate(f)):
        dict_list.append(json.loads(line))
        if i>=max_num_data:
            break

with jsonlines.open('simplified-nq-train.jsonl', 'w') as writer:
    writer.write_all(dict_list)

