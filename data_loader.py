import os
import numpy as np


class ModelData:
    def __init__(self, path):
        class _:
            pass

        self.sets = _()
        self.parameters = _()
        with open(path, "r") as f:
            lines = f.read().strip().split("\n")
        self.sets.L = np.arange(int(lines[0])).astype("int").tolist()
        self.sets.N = np.arange(int(lines[1])).astype("int").tolist()
        self.sets.M = []
        self.parameters.W = []
        self.parameters.S = []
        self.parameters.D = []
        self.parameters.P = []
        self.parameters.A = []
        idx = 2
        for _ in self.sets.N:
            self.sets.M.append(int(lines[idx]))
            idx += 1
            self.parameters.W.append(float(lines[idx]))
            idx += 1
            S_i = []
            D_i = []
            P_i = []
            A_i = []
            for __ in range(self.sets.M[-1]):
                index = 0
                S_i.append(int(lines[idx].split()[index]))
                index += 1
                D_i.append(int(lines[idx].split()[index]))
                index += 1
                P_i.append(int(lines[idx].split()[index]))
                index += 1
                A_i_j = []
                for ___ in range(P_i[-1]):
                    A_i_j.append(int(lines[idx].split()[index]) - 1)
                    index += 1
                A_i.append(A_i_j)
                idx += 1
            self.parameters.S.append(S_i)
            self.parameters.D.append(D_i)
            self.parameters.P.append(P_i)
            self.parameters.A.append(A_i)
        self.parameters.T = int(
            np.sum(np.sum(np.array(self.parameters.D).astype("object")))
        )
        for i in range(len(self.sets.N)):
            paddind_length = max(self.sets.M) - self.sets.M[i]
            for _ in range(paddind_length):
                self.parameters.S[i].append(0)
                self.parameters.D[i].append(0)
                self.parameters.P[i].append(0)
                self.parameters.A[i].append([])


class ModelVars:
    def __init__(self):
        class _:
            pass

        # Decision variables
        self.variables = _()
        self.variables.x = None
        self.variables.y = None
        self.variables.C = None
        self.variables.C_max = None