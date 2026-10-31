"""cand6 canary: prefill-lane stall, grammar termination and throughput checks.

usage: canary.py URL KEYFILE [lanes|grammar|tps|all]
"""

import json
import pathlib
import random
import sys
import threading
import time
import urllib.request

URL, KEY = sys.argv[1], pathlib.Path(sys.argv[2]).read_text().strip()
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
MODEL = "deepseek-v4.1-flash"
WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
]


def post(path, body, timeout):
    req = urllib.request.Request(
        URL + path,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics():
    req = urllib.request.Request(
        URL + "/metrics", headers={"Authorization": f"Bearer {KEY}"}
    )
    out = {}
    with urllib.request.urlopen(req, timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0].removeprefix("vllm:")
            if name in (
                "num_requests_running",
                "num_requests_waiting",
                "generation_tokens_total",
                "prompt_tokens_total",
            ):
                out[name] = out.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return out


def run_parallel(jobs):
    results = [None] * len(jobs)

    def worker(i, fn):
        t0 = time.time()
        try:
            results[i] = ("ok", fn(), time.time() - t0)
        except Exception as e:  # noqa: BLE001
            results[i] = ("err", f"{type(e).__name__}: {e}", time.time() - t0)

    threads = [
        threading.Thread(target=worker, args=(i, fn)) for i, fn in enumerate(jobs)
    ]
    for t in threads:
        t.start()
    return threads, results


def lanes(n=24, lines=4500, timeout=1500):
    """More long prompts than slots: with lanes >= 2 this wedged before #959."""
    rng = random.Random(959)

    def job(i):
        code = f"{rng.randint(1000, 9999)}-{i}"
        body = [
            f"Record {j}: " + " ".join(rng.choice(WORDS) for _ in range(14))
            for j in range(lines)
        ]
        body[rng.randrange(lines)] = f"Record X: the vault code is {code}."
        prompt = (
            "\n".join(body) + "\n\nWhat is the vault code? Answer with just the code."
        )
        return lambda: (
            code,
            post(
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout,
            ),
        )

    t0 = time.time()
    threads, results = run_parallel([job(i) for i in range(n)])
    last_gen, stuck_since, worst_stall = None, None, 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(10)
        try:
            m = metrics()
        except Exception as e:  # noqa: BLE001
            print(f"  metrics error: {e}")
            continue
        progress = m.get("generation_tokens_total", 0) + m.get("prompt_tokens_total", 0)
        busy = m.get("num_requests_running", 0) > 0
        if busy and progress == last_gen:
            stuck_since = stuck_since or time.time()
            worst_stall = max(worst_stall, time.time() - stuck_since)
        else:
            stuck_since = None
        last_gen = progress
        print(
            f"  t={time.time() - t0:5.0f}s"
            f" running={m.get('num_requests_running', 0):.0f}"
            f" waiting={m.get('num_requests_waiting', 0):.0f}"
            f" prompt_tok={m.get('prompt_tokens_total', 0):.0f}"
            f" gen_tok={m.get('generation_tokens_total', 0):.0f}",
            flush=True,
        )
    done = [r for r in results if r[0] == "ok"]
    correct = sum(
        1
        for _, (code, resp), _ in done
        if code in (resp["choices"][0]["message"].get("content") or "")
    )
    prompt_tokens = [resp["usage"]["prompt_tokens"] for _, (_, resp), _ in done]
    for r in results:
        if r[0] == "err":
            print(f"  ERROR {r[1][:200]}")
    print(
        f"LANES done={len(done)}/{n} recall={correct}/{len(done)} "
        f"prompt_tokens~{(sum(prompt_tokens) // max(len(prompt_tokens), 1))} "
        f"worst_no_progress={worst_stall:.0f}s wall={time.time() - t0:.0f}s"
    )
    return len(done) == n and worst_stall < 120


def grammar(n=32, timeout=600):
    """Tool calls and short json_schema answers: grammars terminate mid-draft."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "required": ["city", "unit"],
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                },
            },
        }
    ]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "confidence"],
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 9},
        },
    }
    cities = ["Manila", "Singapore", "Tokyo", "Paris", "Lima", "Oslo", "Cairo", "Seoul"]

    def tool_job(i):
        return lambda: (
            "tool",
            post(
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "max_tokens": 800,
                    "tools": tools,
                    "tool_choice": "required",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"What's the weather in {cities[i % 8]} in celsius?"
                            ),
                        }
                    ],
                },
                timeout,
            ),
        )

    def json_job(i):
        return lambda: (
            "json",
            post(
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "max_tokens": 800,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "yn", "schema": schema, "strict": True},
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Is {i} an even number? Reply in JSON.",
                        }
                    ],
                },
                timeout,
            ),
        )

    jobs = [tool_job(i) if i % 2 else json_job(i) for i in range(n)]
    t0 = time.time()
    threads, results = run_parallel(jobs)
    for t in threads:
        t.join()
    ok = bad = 0
    for r in results:
        if r[0] == "err":
            bad += 1
            print(f"  ERROR {r[1][:200]}")
            continue
        kind, resp = r[1]
        msg = resp["choices"][0]["message"]
        try:
            if kind == "tool":
                call = msg["tool_calls"][0]["function"]
                args = json.loads(call["arguments"])
                assert (
                    call["name"] == "get_weather"
                    and args["unit"] in ("c", "f")
                    and args["city"]
                )
            else:
                obj = json.loads(msg["content"])
                assert set(obj) == {"answer", "confidence"} and obj["answer"] in (
                    "yes",
                    "no",
                )
                assert 0 <= obj["confidence"] <= 9
            ok += 1
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  INVALID {kind}: {type(e).__name__} {str(msg)[:200]}")
    print(f"GRAMMAR valid={ok}/{n} invalid={bad} wall={time.time() - t0:.0f}s")
    return bad == 0


def tps(concurrency, max_tokens=800, timeout=600):
    prompt = "Write a detailed 500-word explanation of how a CPU cache hierarchy works."

    def job():
        return lambda: post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout,
        )

    t0 = time.time()
    threads, results = run_parallel([job() for _ in range(concurrency)])
    for t in threads:
        t.join()
    wall = time.time() - t0
    tokens = sum(r[1]["usage"]["completion_tokens"] for r in results if r[0] == "ok")
    errors = sum(r[0] == "err" for r in results)
    print(
        f"TPS C{concurrency} completion_tokens={tokens} wall={wall:.1f}s "
        f"aggregate={tokens / wall:.1f} tok/s"
        f" per_request={tokens / wall / concurrency:.1f} errors={errors}"
    )
    return errors == 0


if __name__ == "__main__":
    ok = True
    if MODE in ("tps", "all"):
        tps(1, max_tokens=200)  # warm
        ok &= tps(1)
        ok &= tps(8)
    if MODE in ("grammar", "all"):
        ok &= grammar()
    if MODE in ("lanes", "all"):
        ok &= lanes()
    print("CANARY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
