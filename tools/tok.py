#!/usr/bin/env python3
"""Tokenizer side of mikasa (the engine itself takes and returns token ids).
  tok.py encode "<user text>"  -> comma separated ids of the chat-template prompt   (official tokenizer.json + chat_template.jinja)
  tok.py decode a,b,c          -> text (special tokens shown)"""
import os, sys, json
from tokenizers import Tokenizer
from jinja2 import Environment
D = os.environ.get("INKLING_TOK_DIR", os.path.expanduser("~/rrr/inkling-tok"))
tk = Tokenizer.from_file(D + "/tokenizer.json")
if sys.argv[1] == "encode":
    from transformers import AutoTokenizer
    at = AutoTokenizer.from_pretrained(D)
    text = at.apply_chat_template([{"role": "user", "content": sys.argv[2]}], tokenize=False)
    ids = tk.encode(text, add_special_tokens=False).ids
    print(",".join(map(str, ids))); print(repr(text), file=sys.stderr)
else:
    ids = [int(x) for x in sys.argv[2].replace("\n", ",").split(",") if x.strip()]
    print(tk.decode(ids, skip_special_tokens=False))
