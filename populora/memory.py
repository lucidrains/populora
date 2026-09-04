# wrap a recurrent policy so its carried state - the memory - threads through
# a rollout, reset to `init_memory` on each episode start

from __future__ import annotations

import torch
from torch import atleast_1d, cat, nn

from populora._utils import cast_tensor, default, exists

# helpers

def init_memory_tensor(init_memory, num, device = None):
    # normalize the initial memory into a batch of `num` - a scalar broadcasts to
    # every slot, a (1, ...) tensor is expanded, a tensor matching `num` is used as-is.
    # always returns a fresh tensor, never an alias of the caller's

    init = cast_tensor(init_memory, device = device)
    init = atleast_1d(init)

    if init.shape[0] == 1:
        return init.expand(num, *init.shape[1:]).clone()

    assert init.shape[0] == num, f'initial memory must be a scalar or a tensor whose first dim is 1 or {num}, got {tuple(init.shape)}'

    return init.clone()

# main class

class Memory(nn.Module):
    """wrap a recurrent policy so the carried memory threads through a rollout.
    the wrapped network must be a pure function returning (action_logits, mem_next),
    taking the memory as its 1st arg or under `memory_kwarg`"""

    def __init__(
        self,
        net: nn.Module,
        *,
        memory_kwarg: str | None = None,
        init_memory = 0
    ):
        super().__init__()
        self.net = net
        self.memory_kwarg = memory_kwarg

        # registered as a buffer so `.to(device)` moves it along with the network

        self.register_buffer('init_memory', cast_tensor(init_memory))

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.__class__.__name__}(net = {self.net}, memory_kwarg = {self.memory_kwarg}, init_memory = {self.init_memory})'

    def forward(self, mem, obs):
        output = self.net(obs, **{self.memory_kwarg: mem}) if exists(self.memory_kwarg) else self.net(mem, obs)

        assert isinstance(output, (tuple, list)) and len(output) == 2, (
            'a memory-wrapped network must return a 2-tuple (action_logits, mem_next), '
            f'got {type(output)} of length {len(output) if isinstance(output, (tuple, list)) else "-"}'
        )

        return output

def rollout(
    model,
    inputs,
    *,
    individual = None,
    individuals = None,
    all_individuals = False,
    memory = None,
    micro_batch = None
):
    """roll a Memory-wrapped net (or a Population wrapping one) over a batch of
    sequences - one forward per timestep over the whole batch, with the carried
    memory threaded along and reset to `init_memory` at the start of every roll.

    `inputs` are (batch, seq_len, ...) and each timestep feeds the net the slice
    (batch, 1, ...) - scalar features can be passed as a plain (batch, seq_len).
    the per-step outputs are concatenated along the time axis, so single-feature
    nets return (batch, seq_len) and others return (batch, seq_len, ...).

    when given a Population, the forwards are routed exactly like `forward`
    (`all_individuals` tiles every individual across its share of the batch's
    rows, `individual` / `individuals` pin rows to specific ones). otherwise the
    net is rolled directly, and the roll covers a single policy."""

    from populora.population import Population

    if isinstance(model, Population):
        population = model
        net = model.model
        assert isinstance(net, Memory), 'a population must wrap its net in Memory to be rolled out'
    else:
        population = None
        net = model
        assert isinstance(net, Memory), 'rollout expects a Memory-wrapped net, or a Population wrapping one'

    inputs = atleast_1d(inputs)
    assert inputs.ndim >= 2, 'inputs must be (batch, seq_len, ...)'

    batch, seq_len = inputs.shape[0], inputs.shape[1]

    init = default(memory, net.init_memory)
    mem = init_memory_tensor(init, batch, device = inputs.device)

    outputs = []

    with torch.no_grad():
        for t in range(seq_len):
            if exists(population):
                output, mem = population(
                    mem,
                    inputs[:, t:t + 1],
                    individual = individual,
                    individuals = individuals,
                    all_individuals = all_individuals,
                    eval_and_no_grad = True,
                    micro_batch = micro_batch
                )
            else:
                output, mem = net(mem, inputs[:, t:t + 1])

            outputs.append(output)

    return cat(outputs, dim = 1)
