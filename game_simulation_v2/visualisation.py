import numpy as np
import matplotlib.pyplot as plt

def replicator_dynamics_2x2(A, B, resolution=21):
    """
    Affiche le champ de vecteurs de la dynamique réplicative pour un jeu 2x2.
    """
    # Création d'une grille 2D pour x et y
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calcul des variations pour chaque point de la grille
    x_point = X * (1 - X) * (Y * (A[0, 0] - A[1, 0]) + (1 - Y) * (A[0, 1] - A[1, 1]))
    y_point = Y * (1 - Y) * (X * (B[0, 0] - B[0, 1]) + (1 - X) * (B[1, 0] - B[1, 1]))
    
    # Création du champ de vecteurs
    plt.quiver(X, Y, x_point, y_point, color='black', angles='xy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('(Proportion de la population joueur 1 qui joue l\'action A)')
    plt.ylabel('(Proportion de la population joueur 2 qui joue l\'action A)')
    plt.gca().set_aspect('equal', adjustable='box')


