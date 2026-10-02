"""Host-read budgets and numerical equivalence for our eager tensor operations."""

from collections import Counter
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from _runner import run
from sors.training.loss import consistency_js, training_loss


class Operations(TorchDispatchMode):
    """Observe real dispatcher operations; keep all tensor execution unchanged."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.scalar_callers = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.counts[name] += 1
        if func in (torch.ops.aten.index.Tensor, torch.ops.aten.index_put_.default):
            if any(t is not None and t.dtype == torch.bool for t in args[1]):
                self.counts['bool_index'] += 1
        if func is torch.ops.aten._local_scalar_dense.default:
            frame = sys._getframe(1)
            while frame:
                if '/sors/' in frame.f_code.co_filename:
                    self.scalar_callers[(frame.f_code.co_filename, frame.f_code.co_name)] += 1
                    break
                frame = frame.f_back
        return func(*args, **(kwargs or {}))


def _raises_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('invalid inputs were accepted')


def test_consistency_checks_both_views_with_one_scalar_read_and_preserves_gradients():
    torch.manual_seed(8)
    logits = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    slots = torch.tensor([[2, 4, -1], [4, 2, -1], [1, 3, 7], [7, 1, 3]])
    align = torch.tensor([[1, 0, -1], [1, 2, 0]])
    terms = []
    for left, right, ids in ((0, 1, [2, 4]), (2, 3, [1, 3, 7])):
        p, q = logits[left, ids].softmax(0), logits[right, ids].softmax(0)
        middle = (p + q) / 2
        terms.append(((p * (p / middle).log()).sum() + (q * (q / middle).log()).sum()) / 2)
    expected = torch.stack(terms).mean()
    want_grad, = torch.autograd.grad(expected, logits)
    with Operations() as operations:
        actual = consistency_js(logits, slots, align)
    got_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(got_grad, want_grad, atol=1e-15, rtol=1e-13)
    assert operations.counts['aten._local_scalar_dense.default'] == 1, operations.counts
    for row in (0, 1):
        bad = slots.clone()
        bad[row, 1] = -1
        _raises_value_error(lambda: consistency_js(logits, bad, align))


def test_all_slots_membership_uses_fixed_shape_and_preserves_soft_loss_gradients():
    torch.manual_seed(9)
    logits = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    slots = torch.tensor([[2, 4, -1], [4, 2, 6]])
    target = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.2, 0.7]], dtype=torch.float64)
    logp = logits[:, [2, 4, 6, 8]].log_softmax(1)
    expected = -(0.8 * logp[0, 0] + 0.2 * logp[0, 1] +
                 0.1 * logp[1, 1] + 0.2 * logp[1, 0] + 0.7 * logp[1, 2]) / 2
    want_grad, = torch.autograd.grad(expected, logits)
    with Operations() as operations:
        actual = training_loss('all-slots', logits, slots, torch.tensor([0, 2]), [2, 4, 6, 8], target)
    got_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(got_grad, want_grad, atol=1e-15, rtol=1e-13)
    assert operations.counts['bool_index'] == 0, operations.counts
    _raises_value_error(lambda: training_loss('all-slots', logits, slots, torch.tensor([0, 2]), [2, 4, 8], target))


def test_accumulated_metrics_do_not_read_scalars_inside_each_group():
    from test_evaluate import _tiny
    from test_resume import _examples
    from sors.core.batch import collate
    from sors.training.loop import TrainConfig, backward_groups

    for consistency in (0.0, 0.8):
        tok, ids, model = _tiny()
        examples = _examples()
        data = collate(examples, tok, ids, 3)
        cfg = TrainConfig(k_max=3, loss='vocab', consistency=consistency, micro_batches=3)
        with Operations() as operations:
            ce, js, hits, count = backward_groups(model, cfg, examples, data, ids)
        assert isinstance(ce, float) and (js is None if not consistency else isinstance(js, float))
        assert isinstance(hits, int) and isinstance(count, int) and count == len(examples)
        assert 0 <= hits <= count and ce > 0
        reads = {caller: n for caller, n in operations.scalar_callers.items()
                 if caller[1] in ('backward_groups', 'menu_hits')}
        assert not reads, reads


def test_decision_output_preserves_zero_slot_padding_and_gradients_without_bool_indexing():
    from test_decision import tiny_model

    model, _, _, _ = tiny_model()
    slots = torch.tensor([[0, 5, -1, 2], [5, -1, 1, -1]])
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        scores = torch.tensor([[0.1, 0.2, 99, 0.3], [0.4, 98, 0.5, 97]], dtype=dtype, requires_grad=True)
        expected = scores.new_full((2, model.config.vocab_size), float('-inf'))
        for row, column, token in ((0, 0, 0), (0, 1, 5), (0, 3, 2), (1, 0, 5), (1, 2, 1)):
            expected[row, token] = scores[row, column]
        expected = expected.float()
        want_grad, = torch.autograd.grad(-expected.log_softmax(1)[:, 5].sum(), scores)
        with Operations() as operations:
            actual = model._output_logits(scores, slots, slots >= 0)
        got_grad, = torch.autograd.grad(-actual.log_softmax(1)[:, 5].sum(), scores)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(got_grad, want_grad, atol=0, rtol=0)
        assert operations.counts['bool_index'] == 0, operations.counts
        assert operations.counts['aten.nonzero.default'] <= 1, operations.counts
        option_rows = (slots >= 0).flatten().nonzero().flatten()
        with Operations() as reused:
            cached = model._output_logits(scores, slots, slots >= 0, option_rows)
        torch.testing.assert_close(cached, expected, atol=0, rtol=0)
        assert reused.counts['aten.nonzero.default'] == reused.counts['bool_index'] == 0, reused.counts


def test_sparse_embedding_preserves_rows_and_gradients_with_one_dynamic_index():
    from sors.core.model import SlotEmbedding

    torch.manual_seed(4)
    embedding = SlotEmbedding(torch.nn.Embedding(11, 4, dtype=torch.float64), [7, 2])
    for tokens in (torch.tensor([[7, 9, 2, 8, 7, 3], [2, 4, 0, 8, 5, 1]])[:, ::2],
                   torch.tensor([[3, 4, 5]])):
        weights = embedding.base.weight.detach().clone()
        weights[[7, 2]] = embedding.rows
        expected = torch.nn.functional.embedding(tokens, weights)
        with Operations() as operations:
            actual = embedding(tokens)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if 7 in tokens or 2 in tokens:
            want_grad, = torch.autograd.grad(expected.square().sum(), embedding.rows)
            got_grad, = torch.autograd.grad(actual.square().sum(), embedding.rows)
            torch.testing.assert_close(got_grad, want_grad, atol=0, rtol=0)
        else:
            assert not actual.requires_grad
        assert operations.counts['aten._local_scalar_dense.default'] == 0, operations.counts
        assert operations.counts['bool_index'] == 0, operations.counts
        assert operations.counts['aten.nonzero.default'] <= 1, operations.counts


def test_structural_forward_reuses_one_option_selection_and_one_validity_read():
    from test_decision import batch, tiny_model

    model, tok, ids, _ = tiny_model('structural', False, trainable='full')
    data = batch(model, tok, ids)
    with Operations() as operations:
        logits = model.forward_batch(data)
    assert logits.shape == (2, model.config.vocab_size)
    assert operations.counts['bool_index'] == 0, operations.counts
    # Five options in three chunks, plus common text: four embedding calls.
    assert operations.counts['aten.nonzero.default'] <= 5, operations.counts
    validity = sum(n for (path, name), n in operations.scalar_callers.items()
                   if path.endswith('/core/decision.py') and name == '_forward_read_only')
    assert validity == 1, operations.scalar_callers
    for key in ('slot_ids', 'attention_mask'):
        bad = {k: v.clone() for k, v in data.items()}
        bad[key][0] = -1 if key == 'slot_ids' else 0
        _raises_value_error(lambda: model.forward_batch(bad))


def test_repeated_low_precision_embedding_rows_keep_legacy_accumulation_rounding():
    from sors.core.model import SlotEmbedding

    for dtype, small in ((torch.bfloat16, 2**-8), (torch.float16, 2**-11)):
        embedding = SlotEmbedding(torch.nn.Embedding(11, 1, dtype=dtype), [7])
        values = embedding(torch.tensor([[7, 7, 7]]))
        values.backward(torch.tensor([[[1.0], [small], [small]]], dtype=dtype))
        torch.testing.assert_close(embedding.rows.grad, torch.tensor([[1.0]], dtype=dtype), atol=0, rtol=0)


def test_sparse_embedding_accepts_noncontiguous_base_outputs_without_changing_gradients():
    from sors.core.model import SlotEmbedding

    tokens = torch.tensor([[7, 2, 3], [4, 7, 7]])
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        embedding = SlotEmbedding(torch.nn.Embedding(10, 4, dtype=dtype), [7])
        embedding.base.register_forward_hook(
            lambda _module, _args, out: out.transpose(0, 1).contiguous().transpose(0, 1))
        expected = embedding.base(tokens).clone()
        assert not expected.is_contiguous()
        hit = tokens == 7
        expected[hit] = embedding.rows[torch.zeros(int(hit.sum()), dtype=torch.long)]
        upstream = torch.arange(24, dtype=dtype).reshape(2, 3, 4) / 16
        want_grad, = torch.autograd.grad(expected, embedding.rows, upstream)
        actual = embedding(tokens)
        got_grad, = torch.autograd.grad(actual, embedding.rows, upstream)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(got_grad, want_grad, atol=0, rtol=0)


def test_group_reporting_keeps_python_sequential_rounding_exactly():
    from test_evaluate import _tiny
    from test_resume import _examples
    from sors.core.batch import collate, pair_alignment, trim_left_padding
    from sors.core.model import last_logits
    from sors.training.loop import TrainConfig, backward_groups, step_target
    from sors.training.loss import menu_hits

    tok, ids, model = _tiny()
    examples = _examples()
    data = collate(examples, tok, ids, 3)
    cfg = TrainConfig(k_max=3, loss='vocab', consistency=0.8, micro_batches=3)
    target = step_target(examples, data, cfg.label_smoothing)
    alignment = pair_alignment(examples, 3)
    order = torch.argsort(data['attention_mask'].sum(1).reshape(-1, 2).max(1).values,
                          descending=True, stable=True)
    ce_sum, js_sum, hits_sum, n_sum = 0.0, 0.0, 0, 0
    # Legacy oracle: each group reads its scalar and adds to the Python float.
    for units in torch.tensor_split(order, 3):
        rows = (units[:, None] * 2 + torch.arange(2)).flatten()
        logits = last_logits(model, *trim_left_padding(data['input_ids'][rows], data['attention_mask'][rows]))
        slots, gold, targets = data['slot_ids'][rows], data['gold'][rows], data['target'][rows]
        ce = training_loss(cfg.loss, logits, slots, gold, ids, target[rows])
        js = consistency_js(logits, slots, alignment[units])
        weight = len(rows) / len(examples)
        ce_sum += weight * ce.item()
        js_sum += weight * js.item()
        hits, count = menu_hits(logits.detach(), slots, targets)
        hits_sum += hits
        n_sum += count
    got = backward_groups(model, cfg, examples, data, ids)
    assert got == (ce_sum, js_sum, hits_sum, n_sum), (got, (ce_sum, js_sum, hits_sum, n_sum))


def test_output_coordinates_keep_original_bounds_and_avoid_an_extra_full_table_copy():
    from test_decision import tiny_model

    model, _, _, _ = tiny_model()
    scores = torch.tensor([[0.1, 0.2, 99], [0.4, 98, 0.5]], requires_grad=True)
    slots = torch.tensor([[0, 5, -1], [5, -1, 1]])
    with Operations() as operations:
        output = model._output_logits(scores, slots, slots >= 0)
    assert output.is_contiguous()
    assert operations.counts['aten.clone.default'] == 0, operations.counts
    for invalid in (model.config.vocab_size, model.config.vocab_size + 1):
        bad = slots.clone()
        bad[0, 0] = invalid
        try:
            model._output_logits(scores, bad, bad >= 0)
        except (IndexError, RuntimeError):
            continue
        raise AssertionError('out-of-vocabulary coordinate silently disappeared')


if __name__ == '__main__':
    run(globals())
