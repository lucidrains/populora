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

# do the above in a loop
```

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
@misc{fleuret2025freetransformer,
    title   = {The Free Transformer},
    author  = {François Fleuret},
    year    = {2025},
    eprint  = {2510.17558},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2510.17558},
}
```
