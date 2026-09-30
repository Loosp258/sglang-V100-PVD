"""Print causal teacher/Draft token samples from a saved Decode capture."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--captures', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True)
args = parser.parse_args()
records = torch.load(args.captures, map_location='cpu', weights_only=False)['records']
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
for row in records:
    if row['split'] != 'calibration' or row['boundary'] not in (0, 128):
        continue
    print(json.dumps({'id': row['id'], 'boundary': row['boundary'],
                      'committed_decode_tail': tokenizer.decode(row['prefix'][row['prompt_length']:][-32:]),
                      'teacher_future': tokenizer.decode(row['branches']['target']['future']),
                      'draft_future': tokenizer.decode(row['branches']['student']['future'])},
                     ensure_ascii=False), flush=True)
