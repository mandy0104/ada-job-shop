import os
import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Model:

    def __init__(self, model_name, input, output, problem_cnt):
        self.m = gp.Model(model_name)
        self.problem = problem_cnt.split('.')[0]
        self.data = copy.deepcopy(input)
        self.L_cnt = len(self.data.sets.L)
        self.N_cnt = len(self.data.sets.N)
        self.M_cnt = max(self.data.sets.M)
        self.T_cnt = self.data.parameters.T
        self.output = copy.deepcopy(output)
        self.output.variables.C_max = self.m.addVar(vtype=GRB.INTEGER, lb=0.0, ub=float(self.T_cnt))
        self.output.variables.C = self.m.addVars(self.N_cnt, vtype=GRB.INTEGER, lb=0.0, ub=float(self.T_cnt))
        self.output.variables.y = self.m.addVars(self.N_cnt, self.M_cnt, self.T_cnt, vtype=GRB.BINARY, lb=0.0, ub=1.0)

    def pre_solve(self):

        '''
        選所有 jobs 中 W 最大的
        選沒被 block 且 S 夠的當中 D 最小的 operation
        '''

        dp = np.zeros((self.T_cnt, self.L_cnt)).astype('int')
        y = np.zeros((self.N_cnt, self.M_cnt, self.T_cnt)).astype('int')
        finish_time = -1 * np.ones((self.N_cnt, self.M_cnt)).astype('int')

        for t in range(self.T_cnt):
            jobs_idx_sorted_by_w = sorted(zip(np.arange(self.N_cnt).tolist(), self.data.parameters.W), key=lambda x: x[1], reverse=True)
            for job_idx, weight in jobs_idx_sorted_by_w:
                operations_idx_sorted_by_D = sorted(zip(np.arange(self.data.sets.M[job_idx]).tolist(), self.data.parameters.D[job_idx][:self.data.sets.M[job_idx]]), key=lambda x: x[1])
                for operation_idx, d in operations_idx_sorted_by_D:
                    if finish_time[job_idx][operation_idx] == -1 and self.L_cnt - np.sum(dp[t]) >= self.data.parameters.S[job_idx][operation_idx]:
                        needed = self.data.parameters.S[job_idx][operation_idx]
                        usage_queue_if_available = []
                        for l in range(self.L_cnt):
                            if needed and not dp[t][l]:
                                dependency_finished = True
                                for a in self.data.parameters.A[job_idx][operation_idx]:
                                    if finish_time[job_idx][a] == -1 or t <= finish_time[job_idx][a]:
                                        dependency_finished = False
                                if dependency_finished:
                                    needed -= 1
                                    usage_queue_if_available.append(l)
                        if not needed:
                            for t_prime in range(d):
                                for l in usage_queue_if_available:                            
                                    dp[t + t_prime][l] = 1
                                y[job_idx][operation_idx][t + t_prime] = 1
                            finish_time[job_idx][operation_idx] = t + d - 1

        # 同時間資源使用不能超過總資源數
        for t in range(self.T_cnt):
            assert np.sum([self.data.parameters.S[n][m] * y[n][m][t] for n in range(self.N_cnt) for m in range(self.M_cnt)]) <= self.L_cnt

        # 操作必須連續執行
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                assert np.sum([y[n][m][i] * y[n][m][i+1] for i in range(self.T_cnt - 1)]) == max(0, self.data.parameters.D[n][m] - 1)

        # 操作時長等於所需時長
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                assert np.sum([y[n][m][i] for i in range(self.T_cnt)]) == self.data.parameters.D[n][m]

        # Dependency 限制
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                for a in self.data.parameters.A[n][m]:
                    dependency = (np.sum([y[n][a][t] * t for t in range(self.T_cnt)]) * 2 // self.data.parameters.D[n][a] + 1 - self.data.parameters.D[n][a]) // 2
                    target = (np.sum([y[n][m][t] * t for t in range(self.T_cnt)]) * 2 // self.data.parameters.D[n][m] + self.data.parameters.D[n][m] - 1) // 2
                    assert target > dependency


        self.output.variables.C_max.start = np.max(np.where(np.sum(np.sum(y, axis=0), axis=0) != 0)) + 1
        for i, c in enumerate([np.max(np.where(x != 0)) + 1 for x in np.sum(y, axis=1)]):
            self.output.variables.C[i].start = c
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                for t in range(self.T_cnt):
                    self.output.variables.y[n,m,t].start = y[n][m][t]

        # self.T_cnt = np.max(np.where(np.sum(np.sum(y, axis=0), axis=0) != 0)) + 1

        results = []
        for n in range(self.N_cnt):
            for m in range(self.data.sets.M[n]):
                results.append([n, m, int(np.min(np.where(y[n][m] != 0))), self.data.parameters.S[n][m], self.data.parameters.D[n][m], []])
        return results

    def formulation(self):
        '''
        Set Objective
        '''
        self.m.setObjective(
            self.output.variables.C_max +
            gp.quicksum( ( self.data.parameters.W[n] * self.output.variables.C[n] ) for n in range(self.N_cnt) ),
            GRB.MINIMIZE
        )
        print('Objective')
        '''
        Add Constraints
        '''
        # Makespan 的計算
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    for t in range(self.T_cnt):
                        self.m.addConstr(
                            self.output.variables.C[n] >= self.output.variables.y[n,m,t] * t + 1
                        )
        for n in range(self.N_cnt):
            self.m.addConstr(
                self.output.variables.C_max >= self.output.variables.C[n]
            )
        print('c1: Makespan 的計算')
        # 同時間資源使用不能超過總資源數
        for t in range(self.T_cnt):
            self.m.addConstr(
                gp.quicksum( self.data.parameters.S[n][m] * self.output.variables.y[n,m,t] for n in range(self.N_cnt) for m in range(self.M_cnt) if self.data.parameters.S[n][m] ) <= self.L_cnt
            )
        print('c2: 同時間資源使用不能超過總資源數')
        # 操作必須連續執行
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    self.m.addConstr(
                        gp.quicksum(  self.output.variables.y[n,m,t] * self.output.variables.y[n,m,t+1] for t in range(self.T_cnt-1) ) == self.data.parameters.D[n][m] - 1
                    )
        print('c3: 操作必須連續執行')
        # 操作時長等於所需時長
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    self.m.addConstr(
                        gp.quicksum( self.output.variables.y[n,m,t] for t in range(self.T_cnt) ) == self.data.parameters.D[n][m]
                    )
        print('c4: 操作時長等於所需時長')
        # Dependency 限制
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                for m_prime in self.data.parameters.A[n][m]:
                    self.m.addConstr(
                       (gp.quicksum( self.output.variables.y[n,m,t] * t for t in range(self.T_cnt) ) * 2 / self.data.parameters.D[n][m] + 1 - self.data.parameters.D[n][m]) / 2 - 1 >=
                       (gp.quicksum( self.output.variables.y[n,m_prime,t] * t for t in range(self.T_cnt) ) * 2 / self.data.parameters.D[n][m_prime] + self.data.parameters.D[n][m_prime] - 1) / 2
                    )
        print('c5: Dependency 限制')

    def optimize(self, time_limit=600):
        '''
        Optimization
        '''
        # self.m.setParam('Timelimit', time_limit)
        # self.m.params.BestObjStop = 67900.0
        self.m.optimize()

    def dump_results(self):
        results = []
        if self.m.status == GRB.Status.OPTIMAL:
            print('The objective cost: {}'.format(self.m.objVal))
            with open(os.path.join('sol', '{}_sol.txt'.format(self.problem)), 'w') as f:
                f.write('The objective cost: {}\n'.format(self.m.objVal))
                for n in range(self.N_cnt):
                    for m in range(self.M_cnt):
                        if self.data.parameters.S[n][m]:
                            queue = []
                            for t in range(self.T_cnt):
                                if np.round(self.m.getAttr('x', self.output.variables.y)[n,m,t]):
                                    queue.append(t)
                            f.write('Job {} operation {} start\n'.format(n, m))
                            for q in queue:
                                f.write('... {}\n'.format(q))
                            f.write('Job {} operation {} stop\n'.format(n, m))
                            results.append([n, m, queue[0], self.data.parameters.S[n][m], self.data.parameters.D[n][m], []])
                for n in range(self.N_cnt):
                    f.write('Job {} ends at {}\n'.format(n, self.m.getAttr('x', self.output.variables.C)[n]))
                f.write('Makespan is {}\n'.format(self.output.variables.C_max))
        return results