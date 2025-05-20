
import matplotlib.pyplot as plt
from .environnement import Environnement
from .agent import Agent
from .visualisation import replicator_dynamics_2x2
from .games import A_MP, B_MP, A_PD, B_PD, A_A, B_A, A_A2, B_A2

def graphique(A, B, titre, algo):
    for _ in range(10):
        env = Environnement(A, B)
        a1 = Agent("Agent_1", m=1, epsilon=0.2, decay=True, algo=algo)
        a2 = Agent("Agent_2", m=1, epsilon=0.2, decay=True, algo=algo)
        env.ajouter_agents(a1)
        env.ajouter_agents(a2)

        for _ in range(300):
            env.step()

        p1 = a1.histo_probabilities
        p2 = a2.histo_probabilities

        plt.figure(figsize=(6, 6))
        replicator_dynamics_2x2(A, B)
        plt.title(titre)
        plt.plot(p1, p2, 'o-')
        plt.xlabel("P(A) Agent 1")
        plt.ylabel("P(A) Agent 2")
        plt.show()

def main():
    graphique(A_MP, B_MP, "Matching Pennies exp3", "exp3")
    graphique(A_MP, B_MP, "Matching Pennies epsilon", "epsilon")
    graphique(A_PD, B_PD, "Prisoner's Dilemma exp3", "exp3")
    graphique(A_A, B_A, "Autre exp3", "exp3")
    graphique(A_A2, B_A2, "Autre 2 exp3", "exp3")

if __name__ == "__main__":
    main()
