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

pop.save_individual('best.pt', fitnesses.argmax())  # or save just the best individual's weights
```

`pop.evolve_(fitnesses)` runs selection, parent selection, crossover, and mutation in one step. Batch evaluation also supports `pop(x, individuals = [ids])` to route each sample to its own individual.

### Per-weight-matrix operators

`mutation_type`, `epsilon`, and `crossover_type` each accept a dict keyed by LoRA target instead of a scalar — set the mutation rate (or the operator itself) per weight matrix. Keys match a layer's dotted module path or its storage key exactly, else by glob pattern (first match wins); a `'default'` entry catches everything left, coverage must be total, and unmatched patterns raise.

```python
pop.evolve_(
    fitnesses,
    mutation_type = {
        'encoder.*': 'svd_structured',   # glob over the dotted module path
        'default': 'full_gaussian'
    },
    epsilon = {
        'encoder.proj_in': 0.05,         # exact dotted path
        'encoder_proj_out': 0.2,         # exact storage key
        'head': 0.,                      # freeze this matrix entirely
        'default': 0.1
    },
    crossover_type = {
        '*to_q*': 'svd_subspace',
        'default': 'average'
    }
)
```

Any other operator kwarg can vary per target too via the explicit `PerTarget` wrapper: `alpha = PerTarget({'*.proj_in': 5., 'default': 10.})`. Targets sharing the same resolved parameters are dispatched together in one call, so per-target specs cost nothing when they collapse to a single group.

### Fitness sharing

`sharing_quantile` enables Goldberg & Richardson fitness sharing over LoRA genome distance. Each individual's fitness is divided by its niche count — the number of nearby genomes within a radius set by that quantile of all pairwise distances — so a crowded basin cannot crowd out a rare but promising one. It applies to both survivor selection and parent choice, while the raw champion is always retained. The default `0.` leaves selection untouched; `0.25`–`0.5` is the useful range, and `sharing_kernel_power` sharpens the discount (`1.` is the triangular kernel).

```python
pop.evolve_(fitnesses, sharing_quantile = 0.25)
# or, through the environment loop
evolve_with_env(env, model, sharing_quantile = 0.25)
```

`pop.shared_fitnesses(fitnesses, quantile = 0.25)` exposes the discounted scores directly, e.g. for logging.

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

Continuous policies emit actions on `(-1, 1)` — pass `to_range = (-2., 2.)` (or any env action range) to `evolve_with_env` / `interact_with_env` and the actions are rescaled before stepping the env. An action fn built with `make_action` carries its own range, e.g. a beta with `beta_rescale_neg_one_one = False` emits on `(0, 1)` instead — the interactor rescales from that automatically. `interact_with_env(from_range = ...)` is the fallback for custom action functions, and `rescale_from_range_to_range` is exported too.

Rollouts use common random numbers by default: every vector slot is reset with the same per-episode seed sequence (through env-ssl-wrapper's `reset_env` / `reset_env_at`), so fitness differences come from the policies rather than which slot drew an easier initial state.

`standardize = False` skips the running observation normalizer — useful when observations are already bounded (e.g. classic-control) or when the curriculum should see raw states.

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

## Baldwinian selection (test-time training)

`Baldwin` wraps a policy network (or a `Population`) to pair evolutionary selection with live self-predictive test-time training (SPR; Schwarzer et al., 2020) and fast-weight anchor decay (Hinton & Plaut, 1987). Designate which layer emits the latent representation (`latent_layer`) and which layer's LoRA weights act as plastic fast-weight memory (`lora_memory`) — by submodule reference, layer index, or dotted path.

During rollouts, each environment slot carries its own plastic LoRA tensors `(num_envs, d, r)` and `(num_envs, e, r)` on `lora_memory`, while a lightweight `LatentPredictor` learns online to predict the next latent representation from `(z_t, a_t)` and backpropagates into the per-environment LoRAs.

```python
import gymnasium as gym
from torch import nn
from populora import Baldwin, interact_with_env

net = nn.Sequential(
    nn.Linear(8, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 2),
    nn.Tanh()
)

model = Baldwin(
    net,
    latent_layer = net[2],          # submodule, int index, or dotted path
    lora_memory = net[2],           # weight matrix (or matrices) with per-env plastic LoRAs
    carry_across_episodes = True,   # or False to reset plastic LoRAs each episode
    lora_lr = 1.2e-3,
    predictor_lr = 4e-3,
    update_every = 4,
    trust_region = 0.20,
    anchor_decay = 0.03,
)

interactor = interact_with_env(
    gym.make_vec('LunarLanderContinuous-v3', num_envs = 16, enable_wind = True)
)

policy = interactor.evolve(
    model,
    pop_size = 32,
    low_rank = 4,
    num_generations = 20,
    num_episodes = 2,
)
```

Run `uv run train_baldwinian.py` to reproduce the windy LunarLander benchmark and comparison plots across `baseline`, `carry_across_episodes = False`, and `carry_across_episodes = True`.

## Clonal selection (AIS)

`evolve_clonal` borrows from clonal selection in artificial immune systems (de Castro & Von Zuben, 2002). The stagnation-triggered hypermutation burst is the same move as CHC's cataclysmic mutation (Eshelman, 1991). The champion is cloned each generation, its children are hypermutated in LoRA weight space, and the burst restores lost variation before returning to local refinement. The champion is never discarded.

On CartPole with four policies, `bench_cartpole_convergence.py` solves 10/10 seeds in a median of 24 generations, versus 6/10 and a median of 44 for the standard tournament + extrapolative GA.

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

# when it is over, pluck each population's champion

for name in fitnesses:
    coevolve[name].save_individual(f'winners/{name}.pt', fitnesses[name].argmax())
```

Any individual can be saved by index (`save_individual(path, 3)`), loaded back into a population slot (`load_individual`), and the whole population rebuilt with `Population.from_checkpoint`. `demo_coevolve_three.py` runs the same loop with a third population - a `judge` that learns to predict the solver's errors - and shows the winners of all three falling out the same way.

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
@inproceedings{bodnar2020proximal,
    title     = {Proximal Distilled Evolutionary Reinforcement Learning},
    author    = {Cristian Bodnar and Ben Day and Pietro Liò},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {34},
    number    = {04},
    pages     = {3283--3290},
    year      = {2020},
    url       = {https://arxiv.org/abs/1906.09807}
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

```bibtex
@inproceedings{tackett1994unique,
    title     = {The unique implications of brood selection for genetic programming},
    author    = {Tackett, Walter Alden and Carmi, Aviram},
    booktitle = {Proceedings of the First IEEE Conference on Evolutionary Computation. IEEE World Congress on Computational Intelligence},
    pages     = {160--165},
    year      = {1994},
    organization = {IEEE},
    doi       = {10.1109/ICEC.1994.350030}
}
```

```bibtex
@article{decastro2002learning,
    title   = {Learning and Optimization Using the Clonal Selection Principle},
    author  = {de Castro, Leandro N. and Von Zuben, Fernando J.},
    journal = {IEEE Transactions on Evolutionary Computation},
    volume  = {6},
    number  = {3},
    pages   = {239--251},
    year    = {2002}
}
```

```bibtex
@incollection{eshelman1991chc,
    title     = {The CHC Adaptive Search Algorithm: How to Have Safe Search When Engaging in Nontraditional Genetic Recombination},
    author    = {Eshelman, Larry J.},
    booktitle = {Foundations of Genetic Algorithms},
    editor    = {Rawlins, Gregory J. E.},
    volume    = {1},
    pages     = {265--283},
    year      = {1991},
    publisher = {Morgan Kaufmann}
}
```

```bibtex
@inproceedings{schwarzer2020spr,
    title     = {Data-Efficient Reinforcement Learning with Self-Predictive Representations},
    author    = {Max Schwarzer and Ankesh Anand and Rishab Goel and R. Devon Hjelm and Aaron Courville and Philip Bachman},
    booktitle = {International Conference on Learning Representations},
    year      = {2021},
    url       = {https://arxiv.org/abs/2007.05929}
}
```

```bibtex
@misc{teoh2025nextlatentpredictiontransformerslearn,
    title         = {Next-Latent Prediction Transformers Learn Compact World Models},
    author        = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year          = {2025},
    eprint        = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG},
    url           = {https://arxiv.org/abs/2511.05963}
}
```

```bibtex
@inproceedings{hinton1987learning,
    title     = {How Learning Can Guide Evolution},
    author    = {Geoffrey E. Hinton and Steven J. Nowlan},
    booktitle = {Complex Systems},
    volume    = {1},
    pages     = {495--502},
    year      = {1987}
}
```

```bibtex
@misc{bahlousboldi2025dominatednoveltysearch,
    title         = {Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity},
    author        = {Ryan Bahlous-Boldi and Maxence Faldor and Luca Grillotti and Hannah Janmohamed and Lisa Coiffard and Lee Spector and Antoine Cully},
    year          = {2025},
    eprint        = {2502.00593},
    archivePrefix = {arXiv},
    primaryClass  = {cs.NE},
    url           = {https://arxiv.org/abs/2502.00593}
}
```

```bibtex
@article{grillotti2022unsupervised,
    title   = {Unsupervised Behavior Discovery With Quality-Diversity Optimization},
    author  = {Luca Grillotti and Antoine Cully},
    journal = {IEEE Transactions on Evolutionary Computation},
    volume  = {26},
    number  = {6},
    pages   = {1539--1552},
    year    = {2022},
    url     = {https://arxiv.org/abs/2106.05648}
}
```

```bibtex
@inproceedings{cully2019autonomous,
    title     = {Autonomous Skill Discovery with Quality-Diversity and Unsupervised Descriptors},
    author    = {Antoine Cully},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
    pages     = {81--89},
    year      = {2019},
    url       = {https://arxiv.org/abs/1905.11874}
}
```
