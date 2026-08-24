# wrap a recurrent policy so its carried state - the memory - threads through
# a rollout, reset to `init_memory` on each episode start

from __future__ import annotations

from torch import atleast_1d, nn

from populora._utils import cast_tensor, exists

# helpers

def init_memory_tensor(init_memory, num, device = None):
    # normalize the initial memory into a batch of `num` - a scalar broadcasts to
    # every slot, a (1, ...) tensor is expanded, a tensor matching `num` is used as-is.
    # always returns a fresh tensor, never an alias of the caller's

    init = cast_tensor(init_memory, device).to(device)
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
        self.init_memory = init_memory

    def __repr__(self):
        return f'{self.__class__.__name__}(net = {self.net}, memory_kwarg = {self.memory_kwarg}, init_memory = {self.init_memory})'

    def forward(self, mem, obs):
        output = self.net(obs, **{self.memory_kwarg: mem}) if exists(self.memory_kwarg) else self.net(mem, obs)

        assert isinstance(output, (tuple, list)) and len(output) == 2, (
            'a memory-wrapped network must return a 2-tuple (action_logits, mem_next), '
            f'got {type(output)} of length {len(output) if isinstance(output, (tuple, list)) else "-"}'
        )

        return output
