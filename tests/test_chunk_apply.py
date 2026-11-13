"""CPU equivalence of shared chunk views, including non-reentrant checkpointing."""
import copy

import pytest
import torch

from models.heads import chunk_apply, mlp


@pytest.mark.parametrize('checkpointed', [False, True])
def test_split_chunks_preserve_multi_input_outputs_and_gradients(checkpointed):
    torch.manual_seed(34)
    reference = mlp((7, 6, 3), output=True)
    values = (torch.randn(11, 5), torch.randn(11, 2))
    valid = torch.arange(11) % 3 != 0
    upstream = torch.randn(11, 3)
    results = []
    for mode in ('slice', 'split'):
        model = copy.deepcopy(reference)
        x, y = (value.clone().requires_grad_() for value in values)

        def forward(a, b, mask):
            return model(torch.cat((a, b), -1)) * mask[:, None]

        output = chunk_apply(forward, x, y, valid, chunk_size=4,
                             checkpointed=checkpointed, input_mode=mode)
        (output * upstream).sum().backward()
        results.append((output.detach(), x.grad, y.grad,
                        *(p.grad for p in model.parameters())))
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_split_chunks_keep_empty_input_single_call():
    for mode in ('slice', 'split'):
        calls = []
        x, y = torch.empty(0, 3), torch.empty(0, 3)

        def forward(a, b):
            calls.append((a, b))
            return a + b

        output = chunk_apply(forward, x, y, chunk_size=4,
                             checkpointed=True, input_mode=mode)
        assert output.shape == (0, 3)
        assert len(calls) == 1
        assert calls[0][0] is x and calls[0][1] is y
