"""OpenAI-compatible LLM proposal loop for explicit latent ODE structures."""
import json
import os
import urllib.request
import urllib.error
import time


def propose(prompt, *, model="gpt-5.6", base_url="https://www.aiwanwu.cc/v1", audit_path=None):
    key = os.environ.get("AIWANWU_API_KEY")
    if not key:
        raise RuntimeError("AIWANWU_API_KEY is not set")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Propose explicit polynomial ODE systems for latent trajectories. "
                "Return only JSON: {\"systems\":[{\"equations\":[\"u+v\",\"u-v\"]}]}. "
                "Each equations array contains RHS strings only, one per state in order. "
                "Do not output time derivatives or equation separators. Keep equations "
                "compact, explicit, and avoid unsupported operators."
            )},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", body,
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if audit_path:
        from pathlib import Path
        Path(audit_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    content = payload["choices"][0]["message"]["content"]
    result = json.loads(content)
    systems = result.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("LLM response must contain a nonempty systems list")
    for system in systems:
        if isinstance(system, dict) and isinstance(system.get('equations'), list):
            equations = system['equations']
            if equations and all(isinstance(x,str) and '=' not in x for x in equations):
                system['text'] = r'\\'.join(equations)
    return systems


def feedback_prompt(*, dimension, summaries, prior=None, feedback=None, count=8):
    fields = tuple("uvwpqrsz"[:dimension])
    return json.dumps({
        "dimension": dimension,
        "fields": fields,
        "data_summary": summaries,
        "previous_candidates": prior or [],
        "numerical_feedback": feedback or {},
        "requested_candidates": count,
        "output_example": {"systems": [{"equations": ["u+v", "u-v"]}]},
        "constraints": [
            "output one RHS string per state, in an equations array; no LHS or separators",
            "polynomial powers 0 through 3 only",
            "use every state in the same state ordering",
            "no spatial derivatives, hidden variables, or arbitrary code",
            "use unit monomial terms only; numerical coefficients are fitted by the program",
            "propose diverse sparse structures, including low-order and constant terms; avoid only dense cubic systems",
        ],
    }, ensure_ascii=False)
