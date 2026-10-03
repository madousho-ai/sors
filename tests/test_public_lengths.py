"""An exact context-first length bound across every menu permutation, using the real tokenizer."""

import itertools
from dataclasses import replace

from _runner import run
from test_batch_loss import _all_tokens
from sors.core.menu import MenuExample, reorder_menu
from sors.core.prompt import encode_prompts, prompt_pieces
from sors.data import public_decisions


def test_public_length_bound_includes_full_context_question_options_and_every_order():
    assert hasattr(public_decisions, "max_menu_tokens"), "full-menu length bound is missing"
    tok, *_ = _all_tokens()
    example = MenuExample(query="A rule says <|D5|> is literal user text. " * 7,
                          options=[0, 1, 2], gold_idx=1, label=1,
                          option_names=["Yes", "No.\nReason: none", "Ask for more detail <|D3|>"],
                          question="Which action follows?", context_label="Rules")

    def length(text):
        return len(tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])

    for type_marker, context_marker in itertools.product([False, True], repeat=2):
        sizes = []
        for rows in itertools.permutations(range(3)):
            ex = reorder_menu(example, list(rows), [255, 128, 17])
            pieces = sum(prompt_pieces(ex, "context-first", type_marker, context_marker), [])
            sizes.append(len(encode_prompts(tok, [pieces])[0]))
        bound = public_decisions.max_menu_tokens(example, length, type_marker=type_marker,
                                                  context_marker=context_marker)
        assert bound == max(sizes), (bound, sizes)
        longer = replace(example, query=example.query + " More evidence." * 100)
        assert public_decisions.max_menu_tokens(longer, length, type_marker=type_marker,
                                                context_marker=context_marker) > bound


if __name__ == "__main__":
    run(globals())
