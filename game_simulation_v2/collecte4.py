import matplotlib.pyplot as plt
import numpy as np
from .environnement import Environnement
from .agent import Agent
from .visualisation import replicator_dynamics_2x2
from .games import A_MP, B_MP,A_SH,B_SH, A_PD, B_PD, A_A, B_A, A_A2, B_A2

def collecte(A, B, algo1, algo2, n_runs=10000, n_steps=300):
    histo_agent1 = []
    histo_agent2 = []
    histo_r_agent1 = []
    histo_r_agent2 = []

    for _ in range(n_runs):
        env = Environnement(A, B)
        a1 = Agent("Agent_1", m=1, epsilon_init=0.1, decay=True, algo=algo1)
        a2 = Agent("Agent_2", m=1, epsilon_init=0.1, decay=True, algo=algo2)
        env.ajouter_agents(a1)
        env.ajouter_agents(a2)

        for _ in range(n_steps):
            env.step()

        histo_agent1.append(a1.hist_actions)
        histo_agent2.append(a2.hist_actions)
        histo_r_agent1.append(a1.hist_rewards)
        histo_r_agent2.append(a2.hist_rewards)

    return np.array(histo_agent1), np.array(histo_agent2), np.array(histo_r_agent1), np.array(histo_r_agent2)

def plot_action_proportions(histo_actions1, histo_actions2, title):
    """
    histo_actions : np.array de shape (n_runs, n_steps)
    title         : titre du graphique
    """
    # On suppose que les actions sont stockées sous forme de 'A' et 'B' (chaînes)
    prop_A_1 = np.mean(histo_actions1 == 0, axis=0)
    prop_A_2 = np.mean(histo_actions2 == 0, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(prop_A_1, label='Action A agent 1')
    plt.plot(prop_A_2, label='Action A agent 2')
    plt.plot(histo_actions1[0], label='une run')
    plt.xlabel('Étape de simulation')
    plt.ylabel('Proportion')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"action_proportions_{title}.png", dpi=300)


def run_experiment(algo1: str, algo2: str, game: str, MA, MB, tag: str, title: str, 
                   n_runs: int = 100, n_steps: int = 300):
    """Exécute l'expérience, affiche stats, sauve et trace."""
    # 1) Collecte
    histo1, histo2, r1, r2 = collecte(MA, MB, algo1, algo2, n_runs, n_steps)

    # 2) Stats de la récompense par run pour Agent_1
    means = np.mean(r1, axis=1)                  # (n_runs,)
    mean_reward = means.mean()
    std_reward  = means.std(ddof=1)              # écart-type échantillon
    sem_reward  = std_reward / np.sqrt(n_runs)   # SEM

    # 3) Affichage
    print(f"[{tag}] {algo1} vs {algo2}")
    print(f"  • Moyenne récompense Agent 1 : {mean_reward:.4f}")
    print(f"  • Écart-type moyen         : {std_reward:.4f}")
    print(f"  • Erreur-type (SEM)        : {sem_reward:.4f}")

    # 4) Sauvegarde
    np.save(f'histo_{game}_{tag}_v6.npy', histo1)
    np.save(f'histo_r_{game}_{tag}_v6.npy', r1)

    # 5) Plot
    plot_action_proportions(histo1, histo2, game+" "+title)


def main():
    experiments = [

        ("exp3",    "epsilon", "exp3_vs_epsilon", "(exp3 vs ε-greedy)"),
        ("exp3",    "exp3",    "exp3_vs_exp3",    "(exp3 vs exp3)"),
        ("exp3",    "ucb",     "exp3_vs_ucb",     "(exp3 vs UCB)"),
        ("exp3",    "ftl",     "exp3_vs_ftl",     "(exp3 vs FTL)"),
        ("epsilon", "epsilon", "epsilon_vs_epsilon", "(ε-greedy vs ε-greedy)"),
        ("epsilon", "ucb",     "epsilon_vs_ucb",     "(ε-greedy vs UCB)"),
        ("epsilon", "ftl",     "epsilon_vs_ftl",     "(ε-greedy vs FTL)"),
        ("ucb",     "ucb",     "ucb_vs_ucb",         "(UCB vs UCB)"),
        ("ucb",     "ftl",     "ucb_vs_ftl",         "(UCB vs FTL)"),
        ("ftl",     "ftl",     "ftl_vs_ftl",         "(FTL vs FTL)"),
    ]

    for algo1, algo2, tag, title in experiments:
        run_experiment(algo1, algo2,"Matching Pennies",A_MP, B_MP, tag, title)
        run_experiment(algo1, algo2,"Prisoner's Dilemma",A_PD, B_PD, tag, title)
        run_experiment(algo1, algo2,"Stag Hunt",A_SH, B_SH, tag, title)
        run_experiment(algo1, algo2,"Autre",A_A, B_A, tag, title)
        run_experiment(algo1, algo2,"Autre2",A_A2, B_A2, tag, title)
       

if __name__ == "__main__":
    main()
