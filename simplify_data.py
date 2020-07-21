# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# import jsonlines
# from test_utils import simplify_nq_example
# import json


# json_dir = 'v1.0-simplified_nq-dev-all.jsonl'
# dict_list = []
# with open(json_dir) as f:
#     for line in tqdm(f):
#         dict_list.append(simplify_nq_example(json.loads(line)))

# with jsonlines.open('simplified-nq-valid.jsonl', 'w') as writer:
#     writer.write_all(dict_list)
'''Getting simplified data from nq-dev-sample.no-annot.jsonl and nq-dev-sample.jsonl.gz [tiny dataset]'''

import gzip
from test_utils import simplify_nq_example
import jsonlines
import json

# simplify tiny data input samples
tiny_json_input_file = 'tiny-dev/nq-dev-sample.no-annot.jsonl.gz'

tiny_data = [simplify_nq_example(json.loads(line)) for line in gzip.open(tiny_json_input_file, 'r')]

with jsonlines.open('tiny-dev/simplified-dev-sample.no-annot.jsonl', 'w') as f:
  f.write_all(tiny_data)

# simplify tiny data gold labels
tiny_json_gold_file = 'tiny-dev/nq-dev-sample.jsonl.gz'

tiny_d = [simplify_nq_example(json.loads(line)) for line in gzip.open(tiny_json_gold_file, 'r')]

with jsonlines.open('tiny-dev/simplified-dev-sample.jsonl', 'w') as f:
  f.write_all(tiny_d)