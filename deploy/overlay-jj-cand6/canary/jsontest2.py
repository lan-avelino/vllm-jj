"""Strict json_schema yes/no test: validity, correctness, whitespace before answer.

usage: jsontest2.py URL KEYFILE LABEL N
"""

import json
import pathlib
import re
import sys
import threading
import urllib.request

URL, KEY, LABEL, N = (
    sys.argv[1],
    pathlib.Path(sys.argv[2]).read_text().strip(),
    sys.argv[3],
    int(sys.argv[4]),
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "confidence"],
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 9},
    },
}
WS_BEFORE_ANSWER = re.compile(r"^\{\s*\"answer\"\s*:(\s*)")


def post(i):
    body = {
        "model": "deepseek-v4.1-flash",
        "max_tokens": 800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "yn", "schema": SCHEMA, "strict": True},
        },
        "messages": [
            {"role": "user", "content": f"Is {i} an even number? Reply in JSON."}
        ],
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


results = [None] * N


def job(i):
    try:
        results[i] = post(i)
    except Exception as e:
        results[i] = e


for start in range(0, N, 16):
    threads = [
        threading.Thread(target=job, args=(i,))
        for i in range(start, min(start + 16, N))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

valid = correct = errors = 0
ws_lengths, tokens = [], []
for i, r in enumerate(results):
    if isinstance(r, Exception):
        errors += 1
        continue
    content = r["choices"][0]["message"]["content"] or ""
    tokens.append(r["usage"]["completion_tokens"])
    m = WS_BEFORE_ANSWER.match(content)
    ws_lengths.append(len(m.group(1)) if m else -1)
    try:
        answer = json.loads(content)["answer"]
    except Exception:
        continue
    valid += 1
    correct += answer == ("yes" if i % 2 == 0 else "no")

long_ws = sum(1 for w in ws_lengths if w > 2)
print(
    json.dumps(
        {
            "label": LABEL,
            "n": N,
            "errors": errors,
            "valid": valid,
            "correct": correct,
            "ws_gt2": long_ws,
            "ws_max": max(ws_lengths, default=0),
            "tokens_max": max(tokens, default=0),
            "tokens_mean": round(sum(tokens) / max(len(tokens), 1), 1),
        }
    )
)
