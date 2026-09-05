from __future__ import annotations

import math
from typing import Callable

from populora._utils import exists

# helpers

def lerp(start, end, alpha):
    return start + alpha * (end - start)

# base schedule

class Schedule:
    def __init__(self):
        self.step_num = 0

    def reset(self, step = 0):
        self.step_num = step
        return self

    def step(self, step = None):
        if exists(step):
            self.step_num = step

        val = self.get_value(self.step_num)
        self.step_num += 1
        return val

    @property
    def current_value(self):
        return self.get_value(self.step_num)

    def __float__(self):
        return float(self.current_value)

    def __call__(self, step = None):
        if not exists(step):
            return self.step()

        return self.get_value(step)

    def get_value(self, step):
        raise NotImplementedError

# cosine annealing with warm restarts (SGDR)

class CosineAnnealingSchedule(Schedule):
    def __init__(
        self,
        eps_min = 0.02,
        eps_max = 0.18,
        period = 50,
        mult = 1.,
        decay = 1.,
        warmup_generations = 0,
        warm_restart = True
    ):
        super().__init__()
        assert eps_min <= eps_max
        assert period > 0
        assert mult >= 1.

        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.period = int(period)
        self.mult = float(mult)
        self.decay = float(decay)
        self.warmup_generations = int(warmup_generations)
        self.warm_restart = warm_restart

    def get_value(self, step):
        step = max(0, int(step))

        if self.warmup_generations > 0 and step < self.warmup_generations:
            return lerp(self.eps_min, self.eps_max, step / self.warmup_generations)

        t = step - self.warmup_generations

        if self.mult == 1.:
            cycle, t_cur = divmod(t, self.period)
            cur_period = self.period
        else:
            cur_period, cycle = self.period, 0
            while t >= cur_period:
                t -= cur_period
                cur_period = max(1, int(round(cur_period * self.mult)))
                cycle += 1

            t_cur = t

        cur_eps_max = lerp(self.eps_min, self.eps_max, self.decay ** cycle)
        freq = math.pi if self.warm_restart else (2. * math.pi)
        cos_val = math.cos(freq * t_cur / cur_period)

        return lerp(self.eps_min, cur_eps_max, 0.5 * (1. + cos_val))

    def __repr__(self):
        return f'{self.__class__.__name__}(eps_min = {self.eps_min}, eps_max = {self.eps_max}, period = {self.period})'

# oscillating noise schedule

class OscillatingNoiseSchedule(CosineAnnealingSchedule):
    def __init__(
        self,
        eps_min = 0.02,
        eps_max = 0.18,
        period = 50,
        decay = 1.,
        warmup_generations = 0
    ):
        super().__init__(
            eps_min = eps_min,
            eps_max = eps_max,
            period = period,
            mult = 1.,
            decay = decay,
            warmup_generations = warmup_generations,
            warm_restart = False
        )

# linear schedule

class LinearSchedule(Schedule):
    def __init__(
        self,
        eps_start = 0.15,
        eps_end = 0.01,
        num_generations = 100
    ):
        super().__init__()
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.num_generations = max(1, int(num_generations))

    def get_value(self, step):
        step = max(0, int(step))

        if step >= self.num_generations:
            return self.eps_end

        return lerp(self.eps_start, self.eps_end, step / self.num_generations)

    def __repr__(self):
        return f'{self.__class__.__name__}(start = {self.eps_start}, end = {self.eps_end}, gens = {self.num_generations})'

# constant schedule

class ConstantSchedule(Schedule):
    def __init__(self, eps = 0.1):
        super().__init__()
        self.eps = float(eps)

    def get_value(self, step):
        return self.eps

    def __repr__(self):
        return f'{self.__class__.__name__}(eps = {self.eps})'

# helper

ScheduleType = float | int | Schedule | Callable[[int], float]

def as_schedule(schedule_or_value: ScheduleType) -> Schedule | Callable[[int], float]:
    if isinstance(schedule_or_value, (int, float)):
        return ConstantSchedule(float(schedule_or_value))

    if callable(schedule_or_value):
        return schedule_or_value

    raise TypeError(f'expected float, int, Schedule, or callable, got {type(schedule_or_value)}')
