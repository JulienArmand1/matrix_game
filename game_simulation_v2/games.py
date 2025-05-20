import numpy as np

# 1) Prisoner's Dilemma
A_PD = np.array([[3, 0],
                 [5, 1]])
B_PD = np.array([[3, 5],
                 [0, 1]])

# 2) Stag Hunt
A_SH = np.array([[4, 0],
                 [3, 3]])
B_SH = np.array([[4, 3],
                 [0, 3]])

# 3) Matching Pennies
A_MP = np.array([[1, -1],
                 [-1, 1]])
B_MP = -A_MP

# 4) Autre
A_A = np.array([[6, 0],
                 [0, 3]])
B_A = np.array([[6, 15],
                 [0, 1]])

A_A2 = np.array([[6, 1],
                 [0, 8]])
B_A2 = np.array([[6, 0],
                 [1, 8]])


