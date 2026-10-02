"""Official decision subtask adapters. Pure conversion, full menus, hard labels.

Only explicitly supported subsets enter training. Expected subset exclusions are
counted; broken schemas/labels raise ValueError. Gold evidence and future actions
stay outside the model's state. No model calls or automatic downloads.
"""

from __future__ import annotations

import ast
import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from sors.core.menu import MenuExample, reorder_menu
from sors.core.prompt import state_text

PUBLIC_DATASETS = ("contractnli", "maud", "legalbench", "sharc", "conditionalqa", "mind2web", "toolace")
DEFAULT_MANIFEST = pathlib.Path(__file__).resolve().parents[3] / "data/public-decisions/manifest.json"


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty text, got {value!r}")
    return value


def _integer(value, name: str) -> int:
    if type(value) not in (int, str) or not re.fullmatch(r"-?\d+", str(value)):
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    return int(value)


def _menu(names: list[str]) -> None:
    if not isinstance(names, list) or len(names) < 2:
        raise ValueError("menu: expected at least two options")
    for name in names:
        _text(name, "option")
    if len(set(names)) != len(names):
        raise ValueError("menu: duplicate options")


@dataclass
class Decisions:
    examples: list[MenuExample] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    sources: list[dict] = field(default_factory=list)

    def add(self, uid, context, question, names, gold, *, label="", qtype="choice") -> None:
        _menu(names)
        gold = _integer(gold, "answer index")
        if not 0 <= gold < len(names):
            raise ValueError(f"{uid}: answer index {gold} outside menu")
        _text(question, "question")
        if len(names) > 256:
            self.skipped["menu_over_256"] += 1
            return
        self.examples.append(MenuExample(query=state_text(context), options=list(range(len(names))),
                                         gold_idx=gold, label=gold, option_names=list(names),
                                         context_label=label, question=question, qtype=qtype))
        self.ids.append(str(uid))

    def sample(self, n: int, rng: random.Random) -> list[MenuExample]:
        if n < 0 or (n and not self.examples):
            raise ValueError("cannot sample from an empty decision set or use a negative count")
        return [reorder_menu(ex, rng.sample(range(len(ex.options)), len(ex.options)))
                for ex in rng.choices(self.examples, k=n)] if n else []

    def summary(self) -> dict:
        return {"items": len(self.examples), "skipped": dict(self.skipped),
                "menu_sizes": dict(sorted(Counter(len(e.options) for e in self.examples).items())),
                "sources": self.sources}


def contractnli(data: dict) -> Decisions:
    out = Decisions()
    labels = ["Entailment", "Contradiction", "NotMentioned"]
    names = ["Entailment: The contract supports the statement",
             "Contradiction: The contract contradicts the statement",
             "NotMentioned: The contract neither supports nor contradicts the statement"]
    for doc in data["documents"]:
        annotations = doc["annotation_sets"]
        if len(annotations) != 1:
            raise ValueError(f"{doc['id']}: expected one official annotation set")
        for qid, answer in annotations[0]["annotations"].items():
            if qid not in data["labels"] or answer["choice"] not in labels:
                raise ValueError(f"{doc['id']}/{qid}: unknown hypothesis or choice {answer['choice']!r}")
            hypothesis = _text(data["labels"][qid]["hypothesis"], "hypothesis")
            out.add(f"{doc['id']}/{qid}", _text(doc["text"], "contract text"),
                    f"How does the contract relate to this statement?\n{hypothesis}",
                    names, labels.index(answer["choice"]), label="Contract")
    return out


def maud_catalog(text: str) -> dict[str, list[str]]:
    """Read the official QUESTION/ANSWER directory, preserving its complete answer order."""
    catalog = {}
    answers = None
    for line in text.splitlines():
        question = re.fullmatch(r"\s*QUESTION \d+:\s*(.*?)\s*", line)
        answer = re.fullmatch(r"\s*ANSWER (\d+):\s*(.*?)\s*", line)
        if question:
            name = question[1]
            answers = []
            if "[MULTILABEL]" not in name:
                if name in catalog:
                    raise ValueError(f"MAUD directory: duplicate question {name!r}")
                catalog[name] = answers
        elif answer:
            if answers is None or int(answer[1]) != len(answers) + 1:
                raise ValueError("MAUD directory: invalid answer numbering")
            answers.append(answer[2])
    if not catalog:
        raise ValueError("MAUD directory has no single-choice questions")
    for names in catalog.values():
        _menu(names)
    return catalog


def maud(rows: list[dict], catalog: dict[str, list[str]]) -> Decisions:
    """catalog: exact official question -> complete answer list in integer-label order."""
    out = Decisions()
    for i, row in enumerate(rows):
        question = _text(row["question"], "question")
        sub = row.get("subquestion", "").strip()
        gold = _integer(row["label"], "MAUD label")
        if sub and sub != "<NONE>":
            names, qtype = ["no", "yes"], "bool"
            question = f"{question}\nDoes the clause include this answer: {sub}?"
        else:
            if question not in catalog:
                raise ValueError(f"MAUD catalog missing question {question!r}")
            names, qtype = catalog[question], "choice"
            _menu(names)
            if not 0 <= gold < len(names) or names[gold] != row["answer"]:
                raise ValueError(f"MAUD label/answer disagrees with catalog for {question!r}")
        out.add(i, _text(row["text"], "MAUD text"), question, names, gold,
                label="Merger agreement clause", qtype=qtype)
    return out


# Definitions follow the upstream task prompts; no few-shot answers are included.
LEGALBENCH_TASKS = {
    "hearsay": {
        "question": "Is there hearsay?", "context_fields": ["text"], "labels": ["No", "Yes"],
        "rules": "Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted.",
    },
    "abercrombie": {
        "question": "What is the type of mark?", "context_fields": ["text"],
        "labels": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "rules": "A mark is generic if it is the common name for the product. A mark is descriptive if it "
                 "describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests "
                 "or implies a quality or characteristic of the product. A mark is arbitrary if it is a real "
                 "English word that has no relation to the product. A mark is fanciful if it is an invented word.",
    },
    "scalr": {
        "question_field": "question", "context_fields": [],
        "choice_columns": [f"choice_{i}" for i in range(5)], "index_base": 0,
    },
}


def legalbench(rows: list[dict], task: str, spec: dict | None = None) -> Decisions:
    """An explicit task definition selects only input fields, labels and official instructions."""
    spec = LEGALBENCH_TASKS.get(task) if spec is None else spec
    if not spec:
        raise ValueError(f"LegalBench task {task!r} needs an explicit fixed-label/MCQ task spec")
    fields = spec.get("context_fields")
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        raise ValueError("LegalBench task needs context_fields")
    answer_field = spec.get("answer_field", "answer")
    input_fields = fields + spec.get("choice_columns", []) + [spec.get("question_field", "")]
    if {answer_field, "answer", "label", "explanation", "slice"} & set(input_fields):
        raise ValueError("LegalBench answer/annotation fields cannot be input fields")
    if bool(spec.get("labels")) == bool(spec.get("choice_columns")):
        raise ValueError("LegalBench task needs either labels or choice_columns")
    if not (spec.get("question") or spec.get("question_field")):
        raise ValueError("LegalBench task needs a question or question_field")
    if spec.get("index_base", 0) not in (0, 1):
        raise ValueError("LegalBench index_base must be 0 or 1")
    out = Decisions()
    for i, row in enumerate(rows):
        context = {key: row[key] for key in fields}
        if spec.get("rules"):
            context = {"rules": spec["rules"], **context}
        question = row[spec["question_field"]] if spec.get("question_field") else spec["question"]
        if spec.get("labels"):
            names = spec["labels"]
            if row[answer_field] not in names:
                raise ValueError(f"{task}: unknown answer {row[answer_field]!r}")
            gold = names.index(row[answer_field])
        else:
            names = [row[key] for key in spec["choice_columns"]]
            gold = _integer(row[answer_field], "LegalBench answer") - spec.get("index_base", 0)
        out.add(row.get("index", i), context, question, names, gold, label="Legal problem")
    return out


def sharc(rows: list[dict]) -> Decisions:
    out = Decisions()
    names = ["No", "Yes", "Irrelevant: The rule does not address the question",
             "AskMore: Ask a follow-up question to obtain missing information"]
    for i, row in enumerate(rows):
        answer = _text(row["answer"], "ShARC answer").strip().lower()
        gold = {"no": 0, "yes": 1, "irrelevant": 2}.get(answer, 3)
        context = {k: row[k] for k in ("snippet", "scenario", "history")}
        out.add(row.get("utterance_id", i), context, row["question"], names, gold, label="Rule and dialogue")
    return out


def conditionalqa(rows: list[dict], documents: list[dict]) -> Decisions:
    docs = {d["url"]: d for d in documents}
    if len(docs) != len(documents):
        raise ValueError("ConditionalQA: duplicate document URL")
    out = Decisions()
    for row in rows:
        if type(row["not_answerable"]) is not bool:
            raise ValueError("ConditionalQA not_answerable must be boolean")
        answers = row["answers"]
        if row["not_answerable"]:
            out.skipped["unanswerable"] += 1
            continue
        if len(answers) != 1:
            out.skipped["multiple_or_missing_answers"] += 1
            continue
        answer, conditions = answers[0]
        if not isinstance(conditions, list):
            raise ValueError("ConditionalQA answer conditions must be a list")
        if conditions:
            out.skipped["conditional_answer"] += 1
            continue
        if answer not in ("yes", "no"):
            out.skipped["non_boolean_answer"] += 1
            continue
        if row["url"] not in docs:
            raise ValueError(f"ConditionalQA missing document {row['url']!r}")
        doc = docs[row["url"]]
        context = {"title": doc["title"], "document": doc["contents"], "scenario": row["scenario"]}
        out.add(row["id"], context, row["question"], ["no", "yes"], int(answer == "yes"),
                label="Policy and scenario", qtype="bool")
    return out


def mind2web(rows: list[dict]) -> Decisions:
    out = Decisions()
    for record in rows:
        history = record["action_reprs"]
        if len(history) < len(record["actions"]):
            raise ValueError("Mind2Web: incomplete action_reprs history")
        for i, action in enumerate(record["actions"]):
            pos, neg = action["pos_candidates"], action["neg_candidates"]
            if len(pos) != 1:
                out.skipped["multiple_positives" if pos else "no_positive"] += 1
                continue
            candidates = pos + neg
            ids = [str(c["backend_node_id"]) for c in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError("Mind2Web: repeated or conflicting candidate IDs")
            if not neg:
                out.skipped["one_option"] += 1
                continue
            # Candidate order is independent of the positive/negative source lists.
            candidates = sorted(candidates, key=lambda c: str(c["backend_node_id"]))
            names = []
            for c in candidates:
                attributes = c["attributes"]
                if isinstance(attributes, str):
                    attributes = json.loads(attributes)
                names.append(f"{c['backend_node_id']}: {c['tag']} " +
                             json.dumps(attributes, ensure_ascii=False, sort_keys=True))
            gold = next(j for j, c in enumerate(candidates) if str(c["backend_node_id"]) == ids[0])
            context = {"task": record["confirmed_task"], "html": action["cleaned_html"],
                       "previous_actions": history[:i]}
            out.add(f"{record['annotation_id']}/{action.get('action_uid', i)}", context,
                    "Which element should be interacted with next to carry out the task?", names, gold,
                    label="Browser state")
    return out


def _tool_definitions(system: str) -> list[dict] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", system):
        try:
            value, _ = decoder.raw_decode(system[match.start():])
        except ValueError:
            continue
        if isinstance(value, list) and value and all(isinstance(x, dict) and "name" in x for x in value):
            names = [_text(x["name"], "tool name") for x in value]
            if len(set(names)) != len(names):
                raise ValueError("ToolACE: duplicate tool names")
            return value
    return None  # Official data also includes HTML/XML/Markdown and alternate JSON schemas.


def _single_tool(value: str, names: list[str]) -> int | None:
    """Parse the official bracketed-call syntax without executing it, including tool names with spaces."""
    for i, name in enumerate(names):
        match = re.match(r"\[\s*" + re.escape(name) + r"\s*\(", value.strip())
        if not match:
            continue
        source = "[tool(" + value.strip()[match.end():]
        try:
            node = ast.parse(source, mode="eval").body
        except (SyntaxError, ValueError):
            return None
        if not isinstance(node, ast.List) or len(node.elts) != 1:
            return None
        call = node.elts[0]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "tool":
            return None
        # Parameters can contain lists, commas, brackets and JSON true/false/null.
        # Nested function calls and executable expressions are outside the supported subset.
        values = list(call.args) + [kw.value for kw in call.keywords]
        allowed = (ast.Constant, ast.List, ast.Dict, ast.Tuple, ast.Name, ast.Load, ast.UnaryOp, ast.USub, ast.UAdd)
        if any(not isinstance(n, allowed) or (isinstance(n, ast.Name) and n.id not in ("true", "false", "null"))
               for v in values for n in ast.walk(v)):
            return None
        if any(kw.arg is None for kw in call.keywords):
            return None
        return i
    return None


def toolace(rows: list[dict]) -> Decisions:
    out = Decisions()
    for i, record in enumerate(rows):
        tools = _tool_definitions(_text(record["system"], "ToolACE system"))
        if tools is None:
            out.skipped["unsupported_tool_schema_records"] += 1
            continue
        names = [t["name"] for t in tools]
        menu = [f"{t['name']}: {t['description']}" if t.get("description") else t["name"] for t in tools]
        history = []
        for j, message in enumerate(record["conversations"]):
            if message["from"] == "assistant":
                gold = _single_tool(message["value"], names)
                if gold is None:
                    out.skipped["not_single_tool_call"] += 1
                elif len(names) < 2:
                    out.skipped["one_option"] += 1
                else:
                    context = {"tools": tools, "history": list(history)}
                    out.add(f"{i}/{j}", context, "Which tool should be called next?", menu, gold,
                            label="Tools and dialogue")
            history.append({"from": message["from"], "value": message["value"]})
    return out


def _read_source(path: pathlib.Path, files: list[dict], expected: str | None = None):
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: sha256 {digest} != {expected}")
    files.append({"path": str(path), "sha256": digest})
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix in (".csv", ".tsv"):
            return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix == ".json":
            return json.load(f)
    raise ValueError(f"{path}: supported raw formats are JSON, JSONL, CSV and TSV")


def load_public_decisions(name: str, manifest=DEFAULT_MANIFEST) -> Decisions:
    """Read the declared local training sources; dev/test entries are never opened.

    Paths are relative to the manifest. MAUD requires a complete official answer
    catalog; LegalBench task definitions are explicit. Record all files' SHA256
    in the returned summary, including supplemental documents and catalogs.
    """
    if name not in PUBLIC_DATASETS:
        raise ValueError(f"unknown public dataset {name!r}; choose from {PUBLIC_DATASETS}")
    manifest = pathlib.Path(manifest).resolve()
    if not manifest.is_file():
        raise ValueError(f"missing public dataset manifest {manifest}; see public-decisions/README.md in the dataset repository")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("sources"), list):
        raise ValueError("public manifest needs version=1 and a sources list")
    unknown = {s.get("dataset") for s in data["sources"]} - set(PUBLIC_DATASETS)
    if unknown:
        raise ValueError(f"public manifest has unknown dataset(s) {unknown}")
    entries = [s for s in data["sources"] if s["dataset"] == name and s.get("split") == "train"]
    if not entries:
        raise ValueError(f"{name}: no declared train sources in {manifest}")
    out, seen = Decisions(), set()
    for entry in entries:
        path = (manifest.parent / _text(entry.get("path"), "source path")).resolve()
        key = (path, entry.get("task"))
        if key in seen:
            raise ValueError(f"{name}: duplicate source {path}")
        seen.add(key)
        _text(entry.get("source"), "source provenance")
        _text(entry.get("license"), "source license")
        files = []

        def auxiliary(field):
            return _read_source((manifest.parent / _text(entry.get(field), field)).resolve(), files)

        try:
            raw = _read_source(path, files, entry.get("sha256"))
            if name == "contractnli":
                part = contractnli(raw)
            elif name == "maud":
                part = maud(raw, auxiliary("catalog"))
            elif name == "legalbench":
                part = legalbench(raw, _text(entry.get("task"), "LegalBench task"),
                                  auxiliary("task_spec") if entry.get("task_spec") else None)
            elif name == "conditionalqa":
                part = conditionalqa(raw, auxiliary("documents"))
            else:
                part = {"sharc": sharc, "mind2web": mind2web, "toolace": toolace}[name](raw)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{name} {path}: {exc}") from exc
        out.examples.extend(part.examples)
        out.ids.extend(f"{name}/{path.name}/{entry.get('task', '')}/{uid}" for uid in part.ids)
        out.skipped.update(part.skipped)
        out.sources.append({**entry, "files": files, "items": len(part.examples), "skipped": dict(part.skipped)})
    if not out.examples:
        raise ValueError(f"{name}: no eligible training decisions; skipped={dict(out.skipped)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local official-decision training sources (no model/network)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset", required=True, help="Public datasets joined by +")
    args = parser.parse_args()
    names = args.dataset.split("+")
    if len(names) != len(set(names)):
        parser.error("repeated dataset")
    try:
        summaries = {name: load_public_decisions(name, args.manifest).summary() for name in names}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
