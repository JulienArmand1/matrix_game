from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import List

@dataclass
class Agent:
    name: str
    m: int
    epsilon_init: float = 0.1
    decay: bool = True
    algo: str = "epsilon"
    rng: random.Random = field(default_factory=random.Random)

    # État interne
    count: List[int]       = field(init=False, default_factory=lambda: [0, 0])
    reward_sum: List[float]= field(init=False, default_factory=lambda: [0.0, 0.0])
    w: List[float]         = field(init=False, default_factory=lambda: [1.0, 1.0])

    hist_actions: List[int]  = field(init=False, default_factory=list)
    hist_rewards: List[float]= field(init=False, default_factory=list)
    hist_probas:  List[float]= field(init=False, default_factory=list)

    action: int|None = field(init=False, default=None)
    id: int          = field(init=False, default=-1)

    # ---------------- Utils ----------------
    @property
    def total_played(self) -> int:          return sum(self.count)
    
    @property
    def mean(self) -> List[float]:          # moyenne pour chaque bras
        return [
            self.reward_sum[i]/self.count[i] if self.count[i] else 0.
            for i in (0,1)
        ]
    
    @property
    def upper_bound(self) -> List[float]:         
        return [
            1*math.sqrt(2*math.log(self.total_played)/self.count[i])if self.count[i] else 0.
            for i in (0,1)
        ]

    def _ε(self) -> float:
        return self.epsilon_init if not self.decay else \
               self.epsilon_init * (0.9999 ** self.total_played)

    # ---------------- Tirage d'action ----------------
    def play(self) -> int:
        if self.algo == "epsilon":
            action, pA = self._epsilon_greedy()
        elif self.algo == "exp3":
            action, pA = self._exp3()
        elif self.algo == "ucb":
            action, pA = self._ucb()
        elif self.algo == "ftl":
            action, pA = self._ftl()
        else:
            raise ValueError("algo inconnu")

        self.action = action
        self.hist_actions.append(action)
        self.hist_probas.append(pA)
        self.count[action] += 1
        return action

    def _epsilon_greedy(self) -> tuple[int,float]:
        # phase forcée
        for a in (0,1):
            if self.count[a] < self.m:
                return a, 1. if a==0 else 0.
        ε = self._ε()
        if self.rng.random() < ε:                   # exploration
            a = self.rng.randint(0,1)
            return a, 0.5
        # exploitation
        best = 0 if self.mean[0] >= self.mean[1] else 1
        pA   = 1-ε/2 if best==0 else ε/2
        return best, pA

    def _exp3(self) -> tuple[int,float]:
        self.epsilon_init = self._ε()
        total_w = sum(self.w)
        pA = (1-self._ε())*(self.w[0]/total_w) + self._ε()/2
        a  = 0 if self.rng.random() < pA else 1
        return a, pA
    
    def _ftl(self) -> tuple[int,float]:
        # phase forcée
        for a in (0,1):
            if self.count[a] < self.m:
                return a, 1. if a==0 else 0.
            
        best = 0 if self.mean[0]*self.count[0] >= self.mean[1]*self.count[1] else 1
        return best, 1. if best==0 else 0.
    
    def _ucb(self) -> tuple[int,float]:
        # phase forcée
        for a in (0,1):
            if self.count[a] < self.m:
                return a, 1. if a==0 else 0.
        best = 0 if self.mean[0]+self.upper_bound[0] >= self.mean[1]+self.upper_bound[1] else 1
        return best, 1. if best==0 else 0.

    # ---------------- Mise à jour ----------------
    def update(self, reward: float) -> None:
        a = self.action
        self.reward_sum[a] += reward
        self.hist_rewards.append(reward)

        if self.algo == "exp3":
            γ = self.epsilon_init
            p = self.hist_probas[-1] if a==0 else 1-self.hist_probas[-1]
            self.w[a] *= math.exp(γ * reward / (2 * p))
            # normalisation
            s = sum(self.w); self.w[0] /= s; self.w[1] /= s

    def __repr__(self):
        μA, μB = self.mean
        return f"{self.name}: total={self.total_played}, μA={μA:.3f}, μB={μB:.3f}"
