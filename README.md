## PopuLoRA (wip)

Implementation of [PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play](https://arxiv.org/abs/2605.16727v1) (Roger Castanyer et al., [vmax.ai](https://vmax.ai/)). Maintains a population of LoRA adapters on a shared base model, evolves them via selection, crossover, and mutation, and merges the best individual back into the base model.

## Install

```bash
pip install populora
```

## Quick start

```python
import torch
import torch.nn as nn
from populora import Population

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

pop = Population(model, pop_size = 16, low_rank = 4, lora_targets = ['0', '2'])

state = torch.randn(1, 4, 2)
preds = pop(state, all_individuals = True)  # one routed forward over every individual

labels = torch.randn(1, 4, 1)
fitnesses = -((preds - labels) ** 2).reshape(16, -1).mean(dim = -1)

result = pop.select('deterministic', fitnesses, survive_frac = 0.5)

parents = pop.select_parents(
    'tournament',
    fitnesses,
    num_children = len(result.selected_out_indices),
    culled = result.selected_out_indices  # parents come from the survivors
)

pop.crossover_('average', parents, result.selected_out_indices)  # offspring overwrite the culled
pop.mutate_('full_gaussian', individuals = result.selected_out_indices)

model = pop.merge_(fitnesses.argmax())  # merge the best individual back in
```

`pop.evolve_(fitnesses)` runs selection, parent selection, crossover, and mutation in one step. Batch evaluation also supports `pop(x, individuals = [ids])` to route each sample to its own individual.

## Distributed evolution

Each rank evaluates its share of the population and the fitnesses are gathered, so every rank evolves in lockstep. The population is moved to the local device on construction.

```python
import torch
from torch import nn
from populora import Population, is_main_rank

pop = Population(
    nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1)),
    pop_size = 16,
    low_rank = 2,
    lora_targets = ['0', '2']
)

def eval_env(population, idx):
    return population(torch.randn(1, 8), individual = idx).abs().mean().item()

for gen in range(10):
    fitnesses = pop.evaluate_distributed(eval_env)

    if is_main_rank():
        print(f'gen {gen:02d} | best: {fitnesses.max():.3f} | mean: {fitnesses.mean():.3f}')

    pop.evolve_(fitnesses)
```

```bash
torchrun --standalone --nproc-per-node=4 evolve.py
```

Only the LoRA weights are synced across ranks before evaluation (the base model is shared) — pass `sync_base_model = True` to `evaluate_distributed` to broadcast it too.

## Environment evolution

`evolve_with_env` evolves a population against any MDP-style environment in one call — gymnasium, dm_control, isaac, maniskill, pybullet, pufferlib, or any simulator — and returns the merged best policy.

```python
from torch import nn
from dm_control import suite
from populora import evolve_with_env

policy, history = evolve_with_env(
    [suite.load('cartpole', 'balance') for _ in range(16)],  # envs, a list, a vector env, or a factory
    nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2)),
    pop_size = 16,
    low_rank = 16,
    action = lambda logits: logits.argmax(-1) * 2 - 1,
    num_generations = 25,
    horizon = 1000,
    seed = 0,
    progress = True,
    return_history = True  # per-generation best / mean
)
```

`lora_targets` auto-discovers every `Linear` layer when omitted. Pass `target_fitness` to stop early, `checkpoint_dir` to write `latest.pt` (every `checkpoint_every` generations) and `best.pt` (on new bests), and `resume = True` to pick up from the latest checkpoint.

For a custom loop, `interact_with_env` exposes the underlying `EnvInteractor` — one routed forward over the active slots per timestep, distributed across ranks under `torchrun`:

```python
from torch import nn
from dm_control import suite
from populora import interact_with_env

interactor = interact_with_env([suite.load('cartpole', 'balance') for _ in range(16)])

backbone = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))
population = interactor.population(backbone, pop_size = 16, low_rank = 16)

for gen in range(25):
    fitnesses = interactor.evaluate(
        population,
        action = lambda logits: logits.argmax(-1) * 2 - 1,
        horizon = 1000
    )

    population.evolve_(fitnesses)

policy = population.merge_(fitnesses.argmax())
```

Custom fitness functions may take `(population, individuals)` (batched), `(population, idx)` (per index), or `(population)` (all at once) — detected automatically.

## Coevolution

Wrap populations whose fitnesses derive from one another's outputs — e.g. a proposer proposes test inputs while a solver is scored on them, and vice versa. Each population supplies a `probe` (produces its outputs for a step) and a `fitness` (scores it); parameters named after a population receive its outputs, computed once per step in dependency order.

```python
import torch
from torch import nn
from populora import Population, Coevolve, HallOfFame

pop_size = 8
K = 4  # archived solver champions sampled per generation

proposer = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
solver = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

solver_hof = HallOfFame()

def probe_proposer(coevolve):
    return coevolve.proposer(torch.randn(proposer.pop_size, 1), all_individuals = True)  # (P, 1) proposed inputs, one per individual

def probe_solver(coevolve, proposer_outputs):
    return coevolve.solver(proposer_outputs.repeat(solver.pop_size, 1), all_individuals = True)

def fitness_solver(solver_outputs, proposer_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    return -((solver_outputs - target) ** 2).reshape(solver.pop_size, -1).mean(dim = 1)

def fitness_proposer(proposer_outputs, solver_outputs):
    # a proposal earns credit for stumping both the current solver and the archived champions

    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    err = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1).mean(dim = 0)

    arch = solver_hof.probe(solver, proposer_outputs, K)  # (k, P, 1) through sampled champions

    if arch is not None:
        err = err + ((arch - torch.sin(torch.pi * proposer_outputs)) ** 2).mean(dim = (0, 2))

    return err

coevolve = Coevolve(populations = dict(
    proposer = dict(population = proposer, probe = probe_proposer, fitness = fitness_proposer),
    solver = dict(population = solver, probe = probe_solver, fitness = fitness_solver)
))

for gen in range(100):
    fitnesses = coevolve.step()
    solver_hof.add_champion(solver, fitnesses['solver'], generation = gen)
```

`coevolve.history` records the best / mean fitness per population; `coevolve.step(distributed = True)` splits the probes across ranks (round-robin, outputs broadcast raw). Probes must form a chain — a probe depending on its own outputs raises at construction; only fitnesses may close a circle, since every population is probed before any fitness is derived.

## Generation

Autoregressively decode from the population in one batched loop — one routed forward per step over the samples still active.

```python
import torch
from x_transformers import TransformerWrapper, Decoder
from populora import Population, generate

model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = 128,
    attn_layers = Decoder(dim = 128, depth = 2, heads = 2)
)

pop = Population(
    model,
    pop_size = 8,
    low_rank = 4,
    lora_targets = ['attn_layers.layers.0.1.to_q', 'attn_layers.layers.0.1.to_k', 'attn_layers.layers.0.1.to_v']
)

prompts = torch.randint(0, 256, (8, 16))  # one prompt per individual

seqs = generate(
    pop,
    prompts,
    all_individuals = True,
    max_len = 64,
    eos_token = 255,
    cache_kwargs = dict(return_intermediates = True)
)
```

Samples can be routed to explicit individuals (`individual = 3` or `individuals = [...]`, one id per prompt). Early finishers (`eos_token`, or a `stop_fn(tokens, logits, step)`) are compacted out of the batch. Huggingface-style caching works too (`cache_kwarg = 'past_key_values'`, `cache_last_token = True`), and `micro_batch` chunks the routed forward to cap the p-fold activation blowup.

## Citations

```bibtex
@misc{castanyer2026populoracoevolvingllmpopulations,
    title   = {PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play},
    author  = {Roger Creus Castanyer and Geoffrey Bradway and Lorenz Wolf and Maxwill Lin and Augustine N. Mavor-Parker and Matthew James Sargent},
    year    = {2026},
    eprint  = {2605.16727},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2605.16727},
}
```

```bibtex
@misc{schmidhuber2012powerplaytrainingincreasinglygeneral,
    title    = {POWERPLAY: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem},
    author   = {Jürgen Schmidhuber},
    year     = {2012},
    eprint   = {1112.5309},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url      = {https://arxiv.org/abs/1112.5309},
}
```

```bibtex
@misc{xu2026selfimprovinglanguagemodelsbidirectional,
    title   = {Self-Improving Language Models with Bidirectional Evolutionary Search},
    author  = {Guowei Xu and Zhenting Qi and Huangyuan Su and Weirui Ye and Himabindu Lakkaraju and Sham M. Kakade and Yilun Du},
    year    = {2026},
    eprint  = {2605.28814},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2605.28814},
}
```

```bibtex
@misc{bahlousboldi2026vectorpolicyoptimizationtraining,
    title   = {Vector Policy Optimization: Training for Diversity Improves Test-Time Search},
    author  = {Ryan Bahlous-Boldi and Isha Puri and Idan Shenfeld and Akarsh Kumar and Mehul Damani and Sebastian Risi and Omar Khattab and Zhang-Wei Hong and Pulkit Agrawal},
    year    = {2026},
    eprint  = {2605.22817},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2605.22817},
}
```

```bibtex
@misc{bailey2026scalingselfplayselfguidance,
    title   = {Scaling Self-Play with Self-Guidance},
    author  = {Luke Bailey and Kaiyue Wen and Kefan Dong and Tatsunori Hashimoto and Tengyu Ma},
    year    = {2026},
    eprint  = {2604.20209},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2604.20209},
}
```

```bibtex
@misc{petrenko2023dexpbt,
    title    = {DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training},
    author   = {Aleksei Petrenko and Arthur Allshire and Gavriel State and Ankur Handa and Viktor Makoviychuk},
    year     = {2023},
    eprint   = {2305.12127},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO},
    url      = {https://arxiv.org/abs/2305.12127},
}
```
