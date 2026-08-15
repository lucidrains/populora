## PopuLoRA (wip)

Implementation and explorations into [PopuLoRA](https://arxiv.org/abs/2605.16727v1), [Co-Evolving LLM Populations for Reasoning Self-Play](https://vmax.ai/team/populora-co-evolving-llm-populations-for-reasoning-self-play), from Roger Castanyer et al at [vmax.ai](https://vmax.ai/)

## Install

```bash
pip install populora
```

## Usage

```python
import torch
import torch.nn as nn
from populora import Population

# 2-layer MLP

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# wrap with Population

pop = Population(
    model,
    pop_size = 16,
    low_rank = 4,
    lora_targets = ['0', '2']
)

state = torch.randn(1, 4, 2)

# evaluate population against environment

# `individuals` also accepts a list of individual ids (one per sample)

preds = pop(state, all_individuals = True)

labels = torch.randn(1, 4, 1)
fitnesses = -((preds - labels ) ** 2).reshape(16, -1).mean(dim = -1)

# selection

result = pop.select(
    selection_type = 'deterministic',
    fitnesses = fitnesses,
    survive_frac = 0.5
)

# parent selection

parents = pop.select_parents(
    selection_type = 'tournament',
    fitnesses = fitnesses,
    num_children = len(result.selected_out_indices),
    culled = result.selected_out_indices
)

# crossover

pop.crossover_('average', parents, result.selected_out_indices)

# mutate newly generated offspring, preserving surviving elite parents

pop.mutate_('full_gaussian', individuals = result.selected_out_indices)

# alternatively, mutate the entire population

pop.mutate_('full_gaussian', all_individuals = True)

# do the above in a for loop

# ...

# then pick the highest fitness individual and resume RL or fine-tuning on the base model

model = pop.select_and_merge_best_(fitnesses)
```

## Distributed Evolution

Evolution parallelizes trivially - each rank evaluates its share of the population against the environment, the fitnesses are gathered, and the evolution step runs identically on every rank

The population is automatically moved to the distributed device (each rank's local GPU) on construction - pass `device` to `Population` to override. Before the first evaluation, only the LoRA weights are synced across ranks (the base model is shared and identical on every rank) - pass `sync_base_model = True` to `evaluate_distributed` to also broadcast the base model

```python
from time import sleep

import torch
from torch import nn
from populora import Population, is_main_rank

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

pop = Population(
    model,
    pop_size = 16,
    low_rank = 2,
    lora_targets = ['0', '2']
)

x = torch.randn(1, 8)

def eval_env(population, idx):
    sleep(0.1)
    with torch.no_grad():
        # seed the environment with population.eval_seed (shared, auto-synced across ranks)

        return population(x, individual = idx).abs().mean().item() + torch.randn(1).item()

for gen in range(10):

    # distributed evaluation

    fitnesses = pop.evaluate_distributed(eval_env)

    if is_main_rank():
        print(f'gen {gen:02d} | best: {fitnesses.max():.3f} | mean: {fitnesses.mean():.3f}')

    # evolution step

    pop.evolve_(fitnesses)
```

run on 4 processes

```bash
torchrun --standalone --nproc-per-node=4 evolve.py
```

or across machines

```bash
torchrun --nnodes=4 --nproc-per-node=1 --rdzv-endpoint=$MASTER_HOST:29500 evolve.py
```

## Environment Evolution

`evolve_with_env` accepts any environment - from any simulator, vectorized or not, a list of environments, or even an env factory - and evolves a model against it in one call, returning the merged best policy. It leverages [env-ssl-wrapper](https://pypi.org/project/env-ssl-wrapper/) so the same code works for gymnasium, dm_control, isaac, maniskill, pybullet, pufferlib, and anything else that resembles an MDP

```python
from torch import nn
from dm_control import suite
from populora import evolve_with_env

env = suite.load('cartpole', 'balance')  # any env from any sim

# evolve the population in one call - `action` maps the model's outputs to the
# env's actions; `lora_targets` auto-discovers every Linear layer when omitted.
# pass `target_fitness` to stop early once it is reached, and
# `return_history = True` for the per-generation best / mean

policy, history = evolve_with_env(
    [env for _ in range(16)],             # one env, a list, a vector env, or a factory
    nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2)),
    pop_size = 16,
    low_rank = 16,
    action = lambda logits: logits.argmax(-1) * 2 - 1,  # discrete bang-bang
    num_generations = 25,
    horizon = 1000,
    seed = 0,
    progress = True
)
```

`interact_with_env` builds the underlying `EnvInteractor` - each individual plays every environment (or its share of the slots, tiled or strided so all slots stay busy), the population is evaluated in a single routed forward per timestep, and with `torchrun` the individuals are partitioned across ranks automatically - the evolution runs in lockstep everywhere

The underlying pieces are also exposed for custom loops:

```python
from populora import interact_with_env

interactor = interact_with_env(
    [env for _ in range(16)],
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

population = interactor.population(backbone, pop_size = 16, low_rank = 16)

for gen in range(num_generations):
    fitnesses = interactor.evaluate(
        population,
        action = lambda logits: logits.argmax(-1) * 2 - 1,
        horizon = 1000,
        num_episodes = 1
    )

    population.evolve_(fitnesses, survive_frac = 0.5, elite_frac = 0.2, mutation_type = 'full_gaussian', epsilon = 0.2)

policy = population.select_and_merge_best_(fitnesses)
```

Custom fitness functions are accepted in three signatures, detected automatically - `fitness(population, individuals = ...)` (distributed per-rank batch evaluation), `fitness(population, idx)` (per-index), and `fitness(population)` (all at once, evaluated on the main rank and broadcast)

Episodes are seeded deterministically from the shared, auto-synced eval seed, so every run is reproducible and every rank evolves in lockstep

## Coevolution

Wrap multiple populations whose fitnesses derive from one another's outputs - e.g. one population proposes candidates while another judges them, each evolving against the other's current behavior

Each population supplies a `probe` (produces its outputs for a step) and a `fitness` function (scores it). Parameters are injected from the function signature: a parameter named after a population receives that population's outputs (computed once per step, in dependency order)

### Two populations

```python
import torch
from torch import nn
from populora import Population, Coevolve

# the solver fits T(x) = sin(pi x); the proposer proposes test inputs - each is
# scored by the other's outputs

pop_size = 8

proposer = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])
solver = Population(nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

def probe_proposer(coevolve):
    return coevolve.proposer(torch.randn(1, 1), all_individuals = True)  # (P, 1) proposed inputs

def probe_solver(coevolve, proposer_outputs):
    return coevolve.solver(proposer_outputs.repeat(solver.pop_size, 1), all_individuals = True)  # each solver sees all inputs

def fitness_solver(solver_outputs, proposer_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
    return -errors.mean(dim = 1)  # (S,) accuracy on the proposed inputs

def fitness_proposer(proposer_outputs, solver_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
    return errors.mean(dim = 0)  # (P,) error induced on the solver

coevolve = Coevolve(populations = dict(
    proposer = dict(population = proposer, probe = probe_proposer, fitness = fitness_proposer),
    solver = dict(population = solver, probe = probe_solver, fitness = fitness_solver)
))

for _ in range(100):
    coevolve.step()  # probes, derives fitnesses, evolves each population
```

`step` records the best / mean fitness per population in `coevolve.history`; populations are reachable as `coevolve.proposer` / `coevolve['solver']`

### Three populations, in a chain

Append a `judge` that sees every (input, prediction) pair and scores the solver's correctness - the solver must stay accurate while fooling the judge, and the proposer keeps proposing inputs the solver gets wrong

```python
judge = Population(nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)), pop_size = pop_size, low_rank = 2, lora_targets = ['0', '2'])

def probe_judge(coevolve, solver_outputs, proposer_outputs):
    pairs = torch.cat((proposer_outputs.repeat(solver.pop_size, 1), solver_outputs), dim = -1)
    return coevolve.judge(pairs, all_individuals = True)  # (S * P, 1) correctness logits

def fitness_solver(solver_outputs, proposer_outputs, judge_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
    fooled = ((judge_outputs > 0.) & ((solver_outputs - target) ** 2 >= 0.05)).float()  # judge said "correct" on a wrong answer
    return -errors.mean(dim = 1) + 0.25 * fooled.reshape(solver.pop_size, -1).mean(dim = 1)  # accurate and hard to catch

def fitness_judge(solver_outputs, proposer_outputs, judge_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    correct = (solver_outputs - target) ** 2 < 0.05
    acc = ((judge_outputs > 0.) == correct).float().reshape(judge.pop_size, -1).mean(dim = 1)
    return acc  # (J,) how well it catches the solver's mistakes

def fitness_proposer(proposer_outputs, solver_outputs):
    target = torch.sin(torch.pi * proposer_outputs.repeat(solver.pop_size, 1))
    errors = ((solver_outputs - target) ** 2).reshape(solver.pop_size, -1)
    return errors.mean(dim = 0)  # (P,) error its inputs induce on the solver

coevolve = Coevolve(populations = dict(
    proposer = dict(population = proposer, probe = probe_proposer, fitness = fitness_proposer),
    solver = dict(population = solver, probe = probe_solver, fitness = fitness_solver),
    judge = dict(population = judge, probe = probe_judge, fitness = fitness_judge)
))

for _ in range(100):
    coevolve.step(distributed = True)  # distribute the probes across ranks
```

Probes must form a chain - a probe that depends on its own outputs (directly or transitively) raises at construction, reporting the exact cycle (e.g. `proposer -> solver -> proposer`). Fitnesses can close a circle - fitness_A from B's outputs, fitness_B from C's, fitness_C from A's - since every population is probed before any fitness is derived

With `step(distributed = True)`, probes are split across ranks (one per rank, round-robin) and their outputs are broadcast - tensor outputs go over a single raw broadcast, far cheaper than pickling, so each rank derives the same fitnesses and evolves in lockstep. Probes must be pure - only their return value is shared, so side effects (state mutations, logging) happen only on the owning rank and silently diverge across ranks

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
