import numpy as np, random

class Environnement:
    def __init__(self, AA: np.ndarray, BB: np.ndarray):
        assert AA.shape == BB.shape == (2,2)
        self.A, self.B = AA.astype(float), BB.astype(float)
        self.agents, self.agents_count = [], 0

    def ajouter_agents(self, agent):
        self.agents_count += 1
        agent.id = self.agents_count
        self.agents.append(agent)

    def step(self):
        for i in range(0, len(self.agents)-1, 2):
            ag1, ag2 = self.agents[i], self.agents[i+1]
            a1, a2 = ag1.play(), ag2.play()      # entiers 0/1

            ag1.update(self.A[a1,a2])
            ag2.update(self.B[a1,a2])
