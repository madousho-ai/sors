"""Offline fixtures in the official schemas; run with PYTHONPATH=src python tests/test_public_decisions.py."""

import copy
import importlib
import importlib.util
import json
import random
import csv
import hashlib
import pathlib
import subprocess
import sys
import tempfile

from _runner import run
from sors.core.menu import row_alignment, with_partners
from sors.core.prompt import render_menu


def _api():
    name = "sors.data.public_decisions"
    assert importlib.util.find_spec(name) is not None, "official decision adapters are missing"
    return importlib.import_module(name)


def _raises(fn, contains):
    try:
        fn()
    except ValueError as e:
        assert contains in str(e), str(e)
        return
    raise AssertionError(f"expected ValueError containing {contains!r}")


def _contract():
    return {"labels": {"nda-1": {"hypothesis": "The agreement lasts two years."},
                       "nda-2": {"hypothesis": "The agreement permits disclosure."}},
            "documents": [{"id": 4, "text": "The agreement lasts two years.", "spans": [[0, 35]],
                           "annotation_sets": [{"annotations": {
                               "nda-1": {"choice": "Entailment", "spans": [0]},
                               "nda-2": {"choice": "NotMentioned", "spans": []}}}]}]}


def test_contractnli_keeps_all_hypotheses_and_notmentioned_is_a_hard_class():
    out = _api().contractnli(_contract())
    assert len(out.examples) == 2 and len(set(out.ids)) == 2
    a, b = out.examples
    assert a.gold_idx == 0 and b.gold_idx == 2
    assert b.target is None and len(b.options) == 3
    assert a.query == "The agreement lasts two years."
    assert "permits disclosure" in b.question
    assert "annotation_sets" not in render_menu(b)


def test_contractnli_rejects_unknown_labels_and_conflicting_annotators():
    raw = _contract()
    raw["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]["choice"] = "Neutral"
    _raises(lambda: _api().contractnli(raw), "Neutral")
    raw = _contract()
    raw["documents"][0]["annotation_sets"].append({"annotations": {}})
    _raises(lambda: _api().contractnli(raw), "annotation")


def test_maud_uses_the_catalog_even_for_answers_absent_from_this_shard():
    rows = [{"question": "Type of Consideration", "subquestion": "", "text": "Payment is all cash.",
             "answer": "All Cash", "label": "0", "contract_name": "contract_2", "data_type": "main"}]
    catalog = {"Type of Consideration": ["All Cash", "All Stock", "Mixed Cash/Stock", "Mixed Cash/Stock: Election"]}
    out = _api().maud(rows, catalog)
    ex = out.examples[0]
    assert ex.option_names == catalog["Type of Consideration"] and ex.gold_idx == 0
    assert ex.question == "Type of Consideration" and ex.target is None
    _raises(lambda: _api().maud(rows, {}), "catalog")
    rows[0]["label"] = "1"
    _raises(lambda: _api().maud(rows, catalog), "label")


def test_maud_multilabel_subquestions_use_the_binary_label():
    rows = [{"question": "Fundamental representations", "subquestion": "Authority", "text": "A clause.",
             "answer": "Authority, Capitalization", "label": "1"},
            {"question": "Fundamental representations", "subquestion": "Organization", "text": "A clause.",
             "answer": "Authority, Capitalization", "label": "0"}]
    out = _api().maud(rows, {})
    assert [e.gold_idx for e in out.examples] == [1, 0]
    assert all(e.option_names == ["no", "yes"] and e.qtype == "bool" for e in out.examples)
    assert "Authority" in out.examples[0].question and "Organization" in out.examples[1].question


def test_maud_official_none_sentinel_keeps_multiclass_question():
    names = ["All Cash", "All Stock", "Mixed Cash/Stock", "Mixed Cash/Stock: Election"]
    row = {"question": "Type of Consideration-Answer", "subquestion": "<NONE>",
           "text": "Shareholders may elect cash or stock.", "answer": names[3], "label": "3"}
    out = _api().maud([row], {row["question"]: names})
    assert out.examples[0].option_names == names and out.examples[0].gold_idx == 3
    assert "<NONE>" not in out.examples[0].question


def test_maud_official_directory_keeps_all_single_choice_answers_in_order():
    api = _api()
    assert hasattr(api, "maud_catalog"), "official answer-directory parser is missing"
    text = ('## CATEGORY: General Information\nTEXT_TYPE: Consideration\n'
            '\tQUESTION 1: Type of Consideration-Answer\n'
            '\t\tANSWER 1: All Cash\n\t\tANSWER 2: All Stock\n'
            '\t\tANSWER 3: Mixed Cash/Stock\n\t\tANSWER 4: Mixed Cash/Stock: Election\n'
            '\tQUESTION 2: Scope[MULTILABEL]\n\t\tANSWER 1: Authority\n\t\tANSWER 2: Capital\n')
    assert api.maud_catalog(text) == {"Type of Consideration-Answer":
                                     ["All Cash", "All Stock", "Mixed Cash/Stock", "Mixed Cash/Stock: Election"]}
    _raises(lambda: api.maud_catalog(text.replace("ANSWER 2: All Stock", "ANSWER 3: All Stock")), "number")


def test_legalbench_fixed_labels_keep_rules_and_exclude_annotation_fields():
    out = _api().legalbench([{"index": 7, "text": "The witness reported an out-of-court statement.",
                             "answer": "Yes", "slice": "SECRET annotation"}], "hearsay")
    ex = out.examples[0]
    assert ex.option_names == ["No", "Yes"] and ex.gold_idx == 1
    assert "out-of-court" in ex.query and "truth" in ex.query
    assert ex.question == "Is there hearsay?" and "SECRET" not in render_menu(ex)


def test_legalbench_mcq_keeps_zero_based_answer_and_validates_task_spec():
    row = {"index": 0, "question": "Which holding applies?", "answer": 2,
           "choice_0": "Holding A", "choice_1": "Holding B", "choice_2": "Holding C",
           "choice_3": "Holding D", "choice_4": "Holding E"}
    out = _api().legalbench([row], "scalr")
    assert out.examples[0].option_names == ["Holding A", "Holding B", "Holding C", "Holding D", "Holding E"]
    assert out.examples[0].gold_idx == 2 and out.examples[0].question == row["question"]
    _raises(lambda: _api().legalbench([row], "definition_extraction"), "task")
    row["answer"] = 5
    _raises(lambda: _api().legalbench([row], "scalr"), "answer")


def test_legalbench_custom_fixed_task_uses_only_its_explicit_input_fields():
    spec = {"context_fields": ["clause"], "question": "Is this clause enforceable?",
            "labels": ["No", "Yes"], "rules": "Apply the supplied jurisdiction's law."}
    out = _api().legalbench([{"clause": "A contract clause", "answer": "No", "reason": "SECRET"}],
                           "custom", spec)
    assert out.examples[0].gold_idx == 0 and "SECRET" not in render_menu(out.examples[0])
    _raises(lambda: _api().legalbench([], "custom", {**spec, "context_fields": ["answer"]}), "answer")


def test_legalbench_rejects_gold_fields_used_as_question_or_choices():
    base = {"context_fields": ["text"], "question_field": "question", "choice_columns": ["a", "b"]}
    for bad in ({**base, "question_field": "answer"}, {**base, "choice_columns": ["answer", "b"]},
                {**base, "answer_field": "gold", "question_field": "gold"}):
        _raises(lambda: _api().legalbench([], "custom", bad), "answer")


def test_legalbench_explicit_one_based_choices_map_to_zero_based_menu():
    spec = {"context_fields": [], "question_field": "question", "choice_columns": ["a", "b"], "index_base": 1}
    out = _api().legalbench([{"question": "Which?", "a": "Alpha", "b": "Beta", "answer": "2"}], "custom", spec)
    assert out.examples[0].gold_idx == 1
    _raises(lambda: _api().legalbench([], "custom", {**spec, "index_base": 2}), "index_base")


def test_sharc_four_decisions_hide_gold_evidence_and_followup_text():
    rows = [{"utterance_id": str(i), "snippet": "Applicants aged 18 or over are eligible.",
             "question": "Am I eligible?", "scenario": "I am a student.",
             "history": [{"follow_up_question": "Do you live here?", "follow_up_answer": "Yes"}],
             "evidence": [{"follow_up_question": "SECRET EVIDENCE", "follow_up_answer": "Yes"}],
             "answer": answer} for i, answer in enumerate(["Yes", "No", "Irrelevant", "Are you over 18?"])]
    out = _api().sharc(rows)
    assert [e.gold_idx for e in out.examples] == [1, 0, 2, 3]
    for ex in out.examples:
        assert "Do you live here?" in ex.query
        assert "SECRET EVIDENCE" not in render_menu(ex) and "Are you over 18?" not in render_menu(ex)
        assert len(ex.options) == 4 and ex.target is None
    _raises(lambda: _api().sharc([{**rows[0], "answer": ""}]), "answer")


def _conditional_rows():
    base = {"id": "q", "url": "doc-a", "scenario": "I am 20.", "question": "Can I apply?",
            "not_answerable": False, "answers": [["yes", []]], "evidences": ["SECRET evidence"]}
    return [base, {**base, "id": "condition", "answers": [["yes", ["<p>Age at least 18.</p>"]]]},
            {**base, "id": "both", "answers": [["yes", []], ["no", []]]},
            {**base, "id": "span", "answers": [["London", []]]},
            {**base, "id": "missing", "not_answerable": True, "answers": []},
            {**base, "id": "no", "answers": [["no", []]]}]


def test_conditionalqa_keeps_only_unconditional_single_yesno_and_joins_full_document():
    docs = [{"url": "doc-a", "title": "Eligibility", "contents": ["<h1>Rules</h1>", "<p>Age at least 18.</p>"]}]
    out = _api().conditionalqa(_conditional_rows(), docs)
    assert len(out.examples) == 2 and sum(out.skipped.values()) == 4
    assert [e.gold_idx for e in out.examples] == [1, 0]
    assert all(e.qtype == "bool" and e.target is None for e in out.examples)
    assert "Age at least 18" in out.examples[0].query and "I am 20" in out.examples[0].query
    assert "SECRET" not in render_menu(out.examples[0])
    _raises(lambda: _api().conditionalqa(_conditional_rows()[:1], []), "doc-a")


def _element(n, **extra):
    return {"backend_node_id": str(n), "tag": "button", "attributes": json.dumps({"aria-label": f"Button {n}"}),
            **extra}


def _web():
    return {"annotation_id": "web-1", "confirmed_task": "Submit the search form.",
            "action_reprs": ["PAST click", "CURRENT answer", "FUTURE answer"],
            "actions": [{"action_uid": "s0", "cleaned_html": "<main>Start</main>",
                         "operation": {"op": "CLICK", "value": ""},
                         "pos_candidates": [], "neg_candidates": [_element(1)]},
                        {"action_uid": "s1", "cleaned_html": "<main>Search form</main>",
                         "operation": {"op": "CLICK", "value": "SECRET value"},
                         "pos_candidates": [_element(9, is_original_target=True)],
                         "neg_candidates": [_element(2), _element(3)]}]}


def test_mind2web_uses_full_element_menu_and_only_past_actions():
    out = _api().mind2web([_web()])
    assert len(out.examples) == 1 and sum(out.skipped.values()) == 1
    ex = out.examples[0]
    assert len(ex.options) == 3 and "9" in ex.option_names[ex.gold_idx]
    text = render_menu(ex)
    assert "PAST click" in text and "Search form" in text and "Submit the search form" in text
    for hidden in ("CURRENT answer", "FUTURE answer", "SECRET value", "is_original_target", "pos_candidates"):
        assert hidden not in text, hidden


def test_mind2web_skips_multiple_positives_and_oversized_menus_without_sampling_negatives():
    raw = _web()
    action = raw["actions"][1]
    action["pos_candidates"].append(_element(10))
    out = _api().mind2web([raw])
    assert not out.examples and out.skipped["multiple_positives"] == 1
    raw = _web()
    raw["actions"][1]["neg_candidates"] = [_element(n) for n in range(100, 356)]
    out = _api().mind2web([raw])
    assert not out.examples and out.skipped["menu_over_256"] == 1


def test_mind2web_rejects_conflicting_candidate_ids():
    raw = _web()
    raw["actions"][1]["neg_candidates"].append(_element(9))
    _raises(lambda: _api().mind2web([raw]), "candidate")


def _tools():
    tools = [{"name": "Market Trends API", "description": "Market news", "parameters": {"type": "dict"}},
             {"name": "SEC Filings", "description": "Company filings", "parameters": {"type": "dict"}}]
    return {"system": "Here is a list of functions in JSON format that you can invoke:\n" + json.dumps(tools)
                      + '. Return [func1(params), func2(params)].',
            "conversations": [{"from": "user", "value": "Get market news."},
                              {"from": "assistant", "value": '[Market Trends API(country="us")]'},
                              {"from": "tool", "value": "PAST TOOL RESULT"},
                              {"from": "user", "value": "Now get company filings."},
                              {"from": "assistant", "value": '[SEC Filings(identifier="PRIVATE GOLD PARAM")]'},
                              {"from": "tool", "value": "FUTURE TOOL RESULT"}]}


def test_toolace_handles_names_with_spaces_and_uses_only_history_before_the_call():
    out = _api().toolace([_tools()])
    assert len(out.examples) == 2
    a, b = out.examples
    assert a.gold_idx == 0 and b.gold_idx == 1 and all(e.target is None for e in out.examples)
    assert "Get market news" in a.query and "PAST TOOL RESULT" not in a.query
    assert "PAST TOOL RESULT" in b.query
    assert "PRIVATE GOLD PARAM" not in render_menu(b) and "FUTURE TOOL RESULT" not in render_menu(b)
    assert "func1" not in a.query  # source output-format instruction is not part of the state


def test_toolace_filters_parallel_natural_language_nested_calls_and_unknown_tools():
    values = ['[Market Trends API(country="us"), SEC Filings(identifier="x")]',
              'Please provide the ticker.', '[unknown_tool()]',
              '[SEC Filings(identifier=side_effect())]']
    for value in values:
        raw = _tools()
        raw["conversations"] = [raw["conversations"][0], {"from": "assistant", "value": value}]
        out = _api().toolace([raw])
        assert not out.examples and sum(out.skipped.values()) == 1, value


def test_toolace_commas_and_brackets_inside_string_arguments_are_not_parallel_calls():
    raw = _tools()
    raw["conversations"] = [raw["conversations"][0],
                            {"from": "assistant", "value": '[SEC Filings(identifier="a, b [x] (y)")]'}]
    out = _api().toolace([raw])
    assert len(out.examples) == 1 and out.examples[0].gold_idx == 1


def test_toolace_skips_alternative_schema_records_but_keeps_supported_records():
    alternative = {"system": 'Here are the tools you can use:\n{"tool_name":"Stocks","definition":"Stock prices"}',
                   "conversations": [{"from": "user", "value": "Show prices"},
                                     {"from": "assistant", "value": '{Stocks:("page":1)}'}]}
    out = _api().toolace([alternative, _tools()])
    assert len(out.examples) == 2 and out.skipped == {"unsupported_tool_schema_records": 1}
    assert out.ids == ["1/1", "1/4"]


def test_toolace_duplicate_tool_names_still_fail_instead_of_being_filtered():
    raw = _tools()
    raw["system"] = '[{"name":"same"},{"name":"same"}]'
    _raises(lambda: _api().toolace([raw]), "duplicate")


def test_sampling_shuffles_complete_menus_and_keeps_pair_alignment():
    out = _api().contractnli(_contract())
    originals = copy.deepcopy(out.examples)
    batch = out.sample(60, random.Random(5))
    assert out.examples == originals and {e.gold_idx for e in batch} == {0, 1, 2}
    assert all(len(e.options) == 3 and e.options[e.gold_idx] == e.label for e in batch)
    paired = with_partners(batch, random.Random(1))
    for a, b in zip(paired[::2], paired[1::2]):
        assert [b.options[j] for j in row_alignment(a, b)] == a.options
        assert a.option_names[a.gold_idx] == b.option_names[b.gold_idx]


def _fixture(root, dataset="contractnli", data=None, **entry):
    path = root / f"{dataset}.json"
    path.write_text(json.dumps(_contract() if data is None else data), encoding="utf-8")
    source = {"dataset": dataset, "split": "train", "path": path.name,
              "source": "official fixture", "license": "fixture-only", **entry}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "sources": [source]}), encoding="utf-8")
    return manifest, source


def _load(name, path):
    api = _api()
    assert hasattr(api, "load_public_decisions"), "local manifest loader is missing"
    return api.load_public_decisions(name, path)


def test_manifest_reads_only_train_and_records_file_hashes():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, source = _fixture(root)
        manifest.write_text(json.dumps({"version": 1, "sources": [source,
                            {**source, "split": "test", "path": "does-not-exist-test.json"}]}))
        out = _load("contractnli", manifest)
        assert len(out.examples) == 2 and len(out.sources) == 1
        info = out.summary()
        assert info["items"] == 2 and info["menu_sizes"] == {3: 2}
        assert info["sources"][0]["split"] == "train"
        expected = hashlib.sha256((root / "contractnli.json").read_bytes()).hexdigest()
        assert info["sources"][0]["files"][0]["sha256"] == expected


def test_manifest_checks_checksum_and_never_falls_back_to_test():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, sha256="0" * 64)
        _raises(lambda: _load("contractnli", manifest), "sha256")
        manifest, _ = _fixture(root, split="test")
        _raises(lambda: _load("contractnli", manifest), "train")
        _raises(lambda: _load("other", manifest), "dataset")


def test_manifest_requires_provenance_and_rejects_empty_eligible_subset():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, source="")
        _raises(lambda: _load("contractnli", manifest), "source")
        manifest, _ = _fixture(root, "toolace", [_tools()])
        raw = _tools()
        raw["conversations"] = [{"from": "assistant", "value": "Please clarify."}]
        (root / "toolace.json").write_text(json.dumps([raw]))
        _raises(lambda: _load("toolace", manifest), "no eligible")


def test_manifest_reads_maud_csv_and_hashes_the_official_catalog():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, "maud", [], path="MAUD_train.csv", catalog="catalog.json")
        with (root / "MAUD_train.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "subquestion", "text", "answer", "label"])
            writer.writeheader()
            writer.writerow({"question": "Type of Consideration", "subquestion": "", "text": "All cash.",
                             "answer": "All Cash", "label": 0})
        (root / "catalog.json").write_text(json.dumps({"Type of Consideration": ["All Cash", "All Stock"]}))
        out = _load("maud", manifest)
        assert out.examples[0].gold_idx == 0
        assert len(out.sources[0]["files"]) == 2


def test_manifest_reads_legalbench_tsv_and_custom_task_definition():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, "legalbench", [], path="train.tsv", task="custom", task_spec="task.json")
        (root / "train.tsv").write_text('index\ttext\tanswer\n0\tA clause\tYes\n')
        (root / "task.json").write_text(json.dumps({"question": "Does the clause apply?",
                                                    "context_fields": ["text"], "labels": ["No", "Yes"]}))
        out = _load("legalbench", manifest)
        assert out.examples[0].gold_idx == 1 and len(out.sources[0]["files"]) == 2


def test_manifest_reads_conditionalqa_documents_and_jsonl_trajectories():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, "conditionalqa", _conditional_rows(), documents="documents.json")
        (root / "documents.json").write_text(json.dumps([{"url": "doc-a", "title": "Rules", "contents": ["Age 18."]}]))
        out = _load("conditionalqa", manifest)
        assert len(out.examples) == 2 and sum(out.skipped.values()) == 4
        manifest, _ = _fixture(root, "toolace", [], path="train.jsonl")
        (root / "train.jsonl").write_text(json.dumps(_tools()) + "\n\n")
        assert len(_load("toolace", manifest).examples) == 2


def test_manifest_rejects_duplicate_source_entries_instead_of_oversampling():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, source = _fixture(root)
        manifest.write_text(json.dumps({"version": 1, "sources": [source, source]}))
        _raises(lambda: _load("contractnli", manifest), "duplicate")


def test_preflight_command_runs_without_a_model_and_reports_filters():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root)
        # Assert capability before spawning, so the RED phase names the missing behavior.
        _load("contractnli", manifest)
        proc = subprocess.run([sys.executable, "-m", "sors.data.public_decisions",
                               "--manifest", str(manifest), "--dataset", "contractnli"],
                              text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)
        assert result["contractnli"]["items"] == 2 and result["contractnli"]["skipped"] == {}


def test_manifest_allowlist_keeps_complete_examples_and_records_selection_hash():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, include_ids="keep.json")
        (root / "keep.json").write_text(json.dumps(["4/nda-2"]))
        out = _load("contractnli", manifest)
        assert len(out.examples) == 1, "length-selection allowlist was ignored"
        assert out.examples[0].gold_idx == 2 and len(out.examples[0].options) == 3
        assert out.examples[0].query == "The agreement lasts two years."
        assert out.ids == ["contractnli/contractnli.json//4/nda-2"]
        assert out.skipped["excluded_by_include_ids"] == 1
        digest = hashlib.sha256((root / "keep.json").read_bytes()).hexdigest()
        assert out.sources[0]["files"][-1]["sha256"] == digest


def test_manifest_allowlist_rejects_unknown_duplicate_or_nonstring_ids():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, include_ids="keep.json")
        for ids in (["unknown"], ["4/nda-2", "4/nda-2"], [2], {"4/nda-2": True}):
            (root / "keep.json").write_text(json.dumps(ids))
            _raises(lambda: _load("contractnli", manifest), "include_ids")
        (root / "keep.json").write_text("[]")
        _raises(lambda: _load("contractnli", manifest), "no eligible")


def test_manifest_allowlist_verifies_its_declared_checksum():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        manifest, _ = _fixture(root, include_ids="keep.json", include_ids_sha256="0" * 64)
        (root / "keep.json").write_text(json.dumps(["4/nda-2"]))
        _raises(lambda: _load("contractnli", manifest), "sha256")


def _quality():
    return [{"article_id": "article-1", "set_unique_id": "writer-a", "article": "A complete long article.",
             "questions": [{"question_unique_id": "q-a", "question": "Why did she leave?",
                            "options": ["Work", "Travel", "Study", "Family"],
                            "gold_label": 3, "writer_label": 2, "difficult": 1,
                            "validation": [{"secret": "SECRET"}]}]},
            {"article_id": "article-1", "set_unique_id": "writer-b", "article": "A complete long article.",
             "questions": [{"question_unique_id": "q-b", "question": "Where did she go?",
                            "options": ["North", "South", "East", "West"],
                            "gold_label": 1, "writer_label": 1, "difficult": 0}]}]


def test_quality_keeps_both_author_sets_and_uses_validated_one_based_gold():
    assert hasattr(_api(), "quality"), "QuALITY adapter is missing"
    out = _api().quality(_quality())
    assert out.ids == ["q-a", "q-b"] and len(out.examples) == 2
    assert [ex.gold_idx for ex in out.examples] == [2, 0]
    assert all(ex.query == "A complete long article." for ex in out.examples)
    assert "SECRET" not in render_menu(out.examples[0])
    assert all(ex.target is None and ex.qtype == "choice" for ex in out.examples)


def test_quality_rejects_missing_gold_duplicate_ids_and_wrong_option_counts():
    assert hasattr(_api(), "quality"), "QuALITY adapter is missing"
    raw = _quality()
    raw[0]["questions"][0]["gold_label"] = 0
    _raises(lambda: _api().quality(raw), "answer")
    raw = _quality()
    raw[1]["questions"][0]["question_unique_id"] = "q-a"
    _raises(lambda: _api().quality(raw), "duplicate")
    raw = _quality()
    raw[0]["questions"][0]["options"].pop()
    _raises(lambda: _api().quality(raw), "four")


def _reclor():
    return [{"context": "If a permit is valid, work may begin.", "question": "Which follows?",
             "answers": ["A", "B", "C", "D"], "label": 1, "id_string": "train_0"}]


def test_reclor_reads_answers_and_keeps_zero_based_label():
    assert hasattr(_api(), "reclor"), "ReClor adapter is missing"
    out = _api().reclor(_reclor())
    assert out.ids == ["train_0"] and out.examples[0].gold_idx == 1
    assert out.examples[0].option_names == ["A", "B", "C", "D"]
    assert out.examples[0].question == "Which follows?"
    assert out.examples[0].query == "If a permit is valid, work may begin."


def _logiqa2():
    return [{"id": 4554, "answer": 0, "text": "All applicants have permits.",
             "question": "Which follows?", "options": ["A", "B", "C", "D"],
             "type": {"SECRET reasoning label": True}},
            {"id": 4554, "answer": 2, "text": "No applicants have permits.",
             "question": "Which follows?", "options": ["A", "B", "C", "D"], "type": {}}]


def test_logiqa2_preserves_distinct_rows_with_repeated_official_ids():
    assert hasattr(_api(), "logiqa2"), "LogiQA 2.0 adapter is missing"
    out = _api().logiqa2(_logiqa2())
    assert out.ids == ["0/4554", "1/4554"] and len(out.examples) == 2
    assert [ex.gold_idx for ex in out.examples] == [0, 2]
    assert "SECRET" not in render_menu(out.examples[0])


def test_logiqa2_excludes_every_row_of_a_conflicting_gold_group():
    assert hasattr(_api(), "logiqa2"), "LogiQA 2.0 adapter is missing"
    raw = _logiqa2()
    raw += [{**raw[0], "id": 7, "answer": 1}]
    out = _api().logiqa2(raw)
    assert out.ids == ["1/4554"]
    assert out.skipped == {"conflicting_labels": 2}


def test_new_mcq_sources_load_through_manifest_and_preserve_hard_gold():
    for name, raw, expected in (("quality", _quality(), [2, 0]), ("reclor", _reclor(), [1]),
                                ("logiqa2", _logiqa2(), [0, 2])):
        with tempfile.TemporaryDirectory() as d:
            manifest, _ = _fixture(pathlib.Path(d), name, raw)
            out = _load(name, manifest)
            assert [ex.gold_idx for ex in out.examples] == expected
            assert all(ex.target is None for ex in out.examples)


def test_official_mcq_duplicate_options_are_counted_and_excluded_as_whole_questions():
    raw = _quality()
    raw[0]["questions"][0]["options"][1] = raw[0]["questions"][0]["options"][0]
    out = _api().quality(raw)
    assert out.ids == ["q-b"] and out.skipped == {"duplicate_options": 1}
    raw = _reclor()
    raw[0]["answers"] = ["same"] * 4
    out = _api().reclor(raw)
    assert not out.examples and out.skipped == {"duplicate_options": 1}
    raw = _logiqa2()
    raw[0]["options"][2] = raw[0]["options"][3]
    out = _api().logiqa2(raw)
    assert out.ids == ["1/4554"] and out.skipped == {"duplicate_options": 1}


if __name__ == "__main__":
    run(globals())
