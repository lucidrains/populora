from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import Tensor, is_tensor
from torch.nn import Module, ModuleDict

from populora._utils import default, exists, torch_save
from populora.distributed import broadcast_object, distributed_rank, distributed_world_size, is_distributed, preserve_rng
from populora.population import Population, Populations, PopuLoRA

# coevolution - populations whose fitnesses derive from one another's outputs.
# each has a probe (outputs) and a fitness fn; parameters are injected from the
# signature: population names, coevolve, generation / gen, and pop

def _param_deps(fn):
    # the non-default parameter names of a probe / fitness fn

    deps = set()

    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue

        deps.add(name)

    return deps

class Coevolve(Module):
    def __init__(
        self,
        populations: dict[str, Population | dict] | Populations | PopuLoRA,
        *,
        evolve_kwargs: dict[str, dict] | None = None
    ):
        super().__init__()

        if isinstance(populations, (Populations, PopuLoRA)):
            populations = populations.populations

        # a population named like a reserved injection would silently shadow it
        # in every probe / fitness signature

        reserved = self._reserved_names()

        for name in populations:
            assert name not in reserved, f'population name "{name}" collides with a reserved injection - rename the population (reserved: {", ".join(reserved)})'

        self.probes = dict()
        self.fitness_fns = dict()
        self.populations = ModuleDict()

        for name, spec in populations.items():
            if isinstance(spec, dict):
                population = spec['population']
                probe = spec.get('probe')
                fitness = spec.get('fitness')
            else:
                population = spec
                probe = fitness = None

            assert isinstance(population, Population), f'population "{name}" must be a Population or a spec dict'

            self.populations[name] = population

            if exists(probe):
                self.probes[name] = probe
            if exists(fitness):
                self.fitness_fns[name] = fitness

        self.evolve_kwargs = default(evolve_kwargs, dict())

        self.generation = 0
        self.history = {name: [] for name in self.populations}
        self.last_fitnesses = None
        self.last_outputs = None

        # signature introspection is done once per function and cached

        self._plans = dict()
        self._order_cache = dict()

        # fail fast - every probe / fitness parameter must resolve to a population's
        # outputs or a reserved injection

        self._validate(self.probes, 'probe')
        self._validate(self.fitness_fns, 'fitness function')

        # fail fast - probes must form a chain, so any circular probe dependency is
        # caught here at registration time, before the first step ever runs

        self._dependency_order()

    # signature injection

    def _resolve_population(self, name):
        # a parameter names a population's outputs by the population name itself or
        # with an `_outputs` suffix

        if name in self.populations:
            return name

        if name.endswith('_outputs') and name[:-len('_outputs')] in self.populations:
            return name[:-len('_outputs')]

        return None

    @staticmethod
    def _reserved_names():
        return ('coevolve', 'generation', 'gen', 'pop')

    def _plan_param(self, fn, name, param, scope):
        # resolve one parameter to (name, kind, payload...) - kind is one of
        # outputs / coevolve / generation / pop

        resolved = self._resolve_population(name)

        if resolved is not None:
            return (name, 'outputs', resolved)

        if name == 'coevolve':
            return (name, 'coevolve')

        if name in ('generation', 'gen'):
            return (name, 'generation')

        if name == 'pop':
            return (name, 'pop')

        if param.default is not inspect.Parameter.empty:
            return None

        raise self._unresolvable_error(fn, name, scope)

    def _plan(self, fn, scope = 'function'):
        # the resolved parameter plan of a probe / fitness fn, computed once per fn

        plan = self._plans.get(fn)

        if plan is None:
            plan = []

            for name, param in inspect.signature(fn).parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                resolved = self._plan_param(fn, name, param, scope)

                if resolved is not None:
                    plan.append(resolved)

            self._plans[fn] = plan

        return plan

    def _inject_kwargs(self, fn, pop_name, outputs):
        kwargs = dict()

        for name, kind, *rest in self._plan(fn):
            if kind == 'outputs':
                if rest[0] in outputs:
                    kwargs[name] = outputs[rest[0]]
                else:
                    raise self._unresolvable_error(fn, name)
            elif kind == 'coevolve':
                kwargs[name] = self
            elif kind == 'generation':
                kwargs[name] = self.generation
            elif kind == 'pop':
                kwargs[name] = self.populations[pop_name]

        return kwargs

    def _inject(self, fn, pop_name, outputs):
        return fn(**self._inject_kwargs(fn, pop_name, outputs))

    def _validate(self, fns, scope):
        for fn in fns.values():
            self._plan(fn, scope = scope)

    def _unresolvable_error(self, fn, name, scope = 'function'):
        return TypeError(f'cannot resolve parameter "{name}" of {scope} {getattr(fn, "__name__", fn)}')

    # probing - every population's probe runs once per step, in dependency order

    def _dependency_order(self, fitness_fns = None):
        fns = default(fitness_fns, self.fitness_fns)

        key = tuple(id(fn) for fn in fns.values())

        if key in self._order_cache:
            return self._order_cache[key]

        order = []
        visiting = []  # recursion path, so a re-entered population reports the whole cycle

        def compute(name):
            if name in order:
                return

            if name in visiting:
                # a population re-entered while still on the recursion path closes a
                # cycle - report the exact path, not just the re-entered population

                cycle = visiting[visiting.index(name):] + [name]

                raise RuntimeError(
                    f'circular dependency in the probe chain: {" -> ".join(cycle)} - '
                    f'probes must form a chain, since population "{name}" can only be '
                    f'probed from outputs that (transitively) include its own; only '
                    f'fitness functions may close a circle, as they run after every '
                    f'population has been probed'
                )

            visiting.append(name)

            probe = self.probes.get(name)
            assert exists(probe), f'outputs of population "{name}" were requested but it has no probe - add one to its population spec'

            for dep in _param_deps(probe):
                resolved = self._resolve_population(dep)

                if resolved is not None:
                    compute(resolved)

            visiting.pop()
            order.append(name)

        # every population with a probe is probed once per step; anything a fitness
        # function needs from a population without a probe raises at compute time

        for name in self.probes.keys():
            compute(name)

        for fn in fns.values():
            for dep in _param_deps(fn):
                resolved = self._resolve_population(dep)

                if resolved is not None:
                    compute(resolved)

        self._order_cache[key] = order
        return order

    @torch.no_grad()
    def _compute_outputs(self, fitness_fns = None, distributed = False):
        order = self._dependency_order(fitness_fns)

        if distributed:
            if is_distributed():
                return self._compute_outputs_distributed(order)

        outputs = dict()

        for name in order:
            outputs[name] = self._inject(self.probes[name], name, outputs)

        return outputs

    @torch.no_grad()
    def _compute_outputs_distributed(self, order):
        # distributed probes - round-robin ownership in dependency order, outputs
        # broadcast (tensors raw, else pickled) under a preserved rng, so every
        # rank ends with all of them and the evolution stays in sync

        world_size = distributed_world_size()
        outputs = dict()

        with preserve_rng():
            for i, name in enumerate(order):
                owner = i % world_size
                is_owner = distributed_rank() == owner
                value = self._inject(self.probes[name], name, outputs) if is_owner else None
                outputs[name] = broadcast_object(value, src = owner)

        return outputs

    # stepping

    @torch.no_grad()
    def evolve_(
        self,
        fitnesses: dict[str, Tensor],
        evolve_kwargs: dict[str, dict] | None = None
    ):
        # per-call kwargs override the constructor's per population, merging
        # name-by-name - a call that tunes one population must not silently
        # drop the constructor settings of the others

        ctor_kwargs = self.evolve_kwargs
        call_kwargs = default(evolve_kwargs, dict())

        merged = dict(ctor_kwargs)

        for name, kwargs in call_kwargs.items():
            base = merged.get(name)
            merged[name] = {**base, **kwargs} if isinstance(base, dict) and isinstance(kwargs, dict) else kwargs

        for name, pop in self.populations.items():
            assert name in fitnesses, f'fitnesses missing for population {name}'

            f = fitnesses[name]
            assert f.ndim == 1 and f.shape[0] == pop.pop_size, f'fitnesses for {name} must be of shape ({pop.pop_size},)'

            pop.evolve_(f.to(pop.device), **merged.get(name, dict()))

        self.generation += 1
        return self

    @torch.no_grad()
    def step_(
        self,
        fitness_fns: dict[str, callable] | None = None,
        evolve_kwargs: dict[str, dict] | None = None,
        distributed: bool = False
    ):
        # probe every population, derive each population's fitness, and evolve them
        # all - `distributed` distributes the probes across ranks by population

        fns = dict(self.fitness_fns)
        fns.update(default(fitness_fns, dict()))

        self._validate(fns, 'fitness function')

        outputs = self._compute_outputs(fitness_fns = fns, distributed = distributed)
        self.last_outputs = outputs

        fitnesses = dict()

        for name, pop in self.populations.items():
            fn = fns.get(name)
            assert exists(fn), f'no fitness function for population "{name}" - add one to its population spec or pass it to step_'

            f = self._inject(fn, name, outputs)
            assert is_tensor(f), f'fitness function for population "{name}" must return a tensor, got {type(f)}'
            assert f.ndim == 1 and f.shape[0] == pop.pop_size, f'fitness for "{name}" must be of shape ({pop.pop_size},), got {tuple(f.shape)}'

            fitnesses[name] = f

        self.last_fitnesses = fitnesses

        self.evolve_(fitnesses, evolve_kwargs = evolve_kwargs)

        for name, f in fitnesses.items():
            self.history[name].append((f.max().item(), f.mean().item()))

        return fitnesses

    step = step_

    # convenience access - coevolve.proposer, coevolve['proposer']

    def __getattr__(self, name):
        # populations are reachable by attribute, e.g. coevolve.proposer

        populations = self.__dict__.get('_modules', dict()).get('populations')

        if populations is not None and name in populations:
            return populations[name]

        return super().__getattr__(name)

    def forward(self, pop_name: str, *args, individual = None, individuals = None, all_individuals: bool | None = None, **kwargs):
        # route through a population - by default over all individuals, so
        # cross-population fitness can be derived from batched outputs

        assert pop_name in self.populations, f'unknown population {pop_name}'

        if all_individuals is None:
            all_individuals = not (exists(individual) or exists(individuals))

        return self.populations[pop_name](*args, individual = individual, individuals = individuals, all_individuals = all_individuals, **kwargs)

    def __getitem__(self, pop_name: str):
        return self.populations[pop_name]

    def __len__(self):
        return len(self.populations)

    @property
    def device(self):
        return next(self.parameters()).device

    # save and load

    @torch.no_grad()
    def state_dict_pkg(self, save_base_model: bool = True):
        pkg = {name: pop.state_dict_pkg(save_base_model = save_base_model) for name, pop in self.populations.items()}
        pkg['_generation'] = self.generation
        pkg['_history'] = self.history
        return pkg

    @torch.no_grad()
    def save(self, path: str | Path, save_base_model: bool = True):
        torch_save(self.state_dict_pkg(save_base_model = save_base_model), path)
        return self

    @torch.no_grad()
    def load(self, path: str | Path | dict, strict: bool = True):
        pkg = torch.load(path, map_location = self.device, weights_only = False) if not isinstance(path, dict) else path

        for name, pop in self.populations.items():
            if name in pkg:
                pop.load(pkg[name], strict = strict)

        if '_generation' in pkg:
            self.generation = pkg['_generation']
        if '_history' in pkg:
            self.history = pkg['_history']

        return self
