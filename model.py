import os
import copy
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def save_checkpoint(model, where):
    try:
        model_check_point = np.array([abs(var.x) for var in model.getVars()])
        np.save(os.path.join("sol", model.ModelName), model_check_point)
    except:
        pass


class Model:
    def __init__(self, model_name, input, output, problem_cnt):
        print("Creating model: {}".format(model_name))
        self.m = gp.Model(name=model_name)
        self.problem = problem_cnt.split(".")[0]
        self.data = copy.deepcopy(input)
        self.L_cnt = len(self.data.sets.L)
        self.N_cnt = len(self.data.sets.N)
        self.M_cnt = max(self.data.sets.M)
        self.T_cnt = self.data.parameters.T
        self.output = copy.deepcopy(output)

    def __cal_obj_numpy(self, x):
        return (
            np.max(np.max(x + self.data.parameters.D - 1, axis=1))
            + np.sum(
                self.data.parameters.W * np.max(x + self.data.parameters.D - 1, axis=1)
            )
            * self.window
        )

    def __get_sol_result_params(self, path):
        try:
            saved_model_params = np.load(path)
            x_saved = np.empty((self.N_cnt, self.M_cnt)).astype("int")
            y_saved = np.zeros((self.N_cnt, self.M_cnt, self.T_cnt)).astype("int")
            tmp_T_cnt = (
                int(
                    (len(saved_model_params) - (1 + self.N_cnt))
                    / self.N_cnt
                    / self.M_cnt
                )
                - 1
            )
            npy_idx = 1 + self.N_cnt
            for n in range(self.N_cnt):
                for m in range(self.M_cnt):
                    x_saved[n][m] = saved_model_params[npy_idx]
                    npy_idx += 1
            for n in range(self.N_cnt):
                for m in range(self.M_cnt):
                    for t in range(tmp_T_cnt):
                        y_saved[n][m][t] = saved_model_params[npy_idx]
                        npy_idx += 1
            return x_saved, y_saved
        except:
            return None, None

    def gen_operations_order(self, problem_prefix):
        x_saved, _ = self.__get_sol_result_params(
            os.path.join("sol", "{}.sol.npy".format(problem_prefix))
        )
        results = []
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    results.append(
                        [
                            n,
                            m,
                            x_saved[n][m],
                            self.data.parameters.S[n][m],
                            self.data.parameters.D[n][m],
                            [],
                        ]
                    )
        return results

    def pre_solve(self, window):

        print("Running presolve...")
        print("window = {}".format(window))

        self.window = window
        self.data.parameters.D = (
            np.ceil(np.array(self.data.parameters.D) / window).astype("int").tolist()
        )

        """
        選所有 jobs 中 W 最大的
        選沒被 block 且 S 夠的當中 D 最小的 operation
        """

        dp = np.zeros((self.T_cnt, self.L_cnt)).astype("int")
        x = np.empty((self.N_cnt, self.M_cnt))
        y = np.zeros((self.N_cnt, self.M_cnt, self.T_cnt)).astype("int")
        finish_time = -1 * np.ones((self.N_cnt, self.M_cnt)).astype("int")

        for t in range(self.T_cnt):
            jobs_idx_sorted_by_w = sorted(
                zip(
                    np.arange(self.N_cnt).tolist(),
                    self.data.parameters.W,
                    np.sum(self.data.parameters.D, axis=1).tolist(),
                    np.sum(self.data.parameters.S, axis=1).tolist(),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            for job_idx, _, __, ___ in jobs_idx_sorted_by_w:
                operations_idx_sorted_by_D = sorted(
                    zip(
                        np.arange(self.data.sets.M[job_idx]).tolist(),
                        self.data.parameters.D[job_idx][: self.data.sets.M[job_idx]],
                    ),
                    key=lambda x: x[1],
                )
                for operation_idx, d in operations_idx_sorted_by_D:
                    if (
                        finish_time[job_idx][operation_idx] == -1
                        and self.L_cnt - np.sum(dp[t])
                        >= self.data.parameters.S[job_idx][operation_idx]
                    ):
                        needed = self.data.parameters.S[job_idx][operation_idx]
                        usage_queue_if_available = []
                        for l in range(self.L_cnt):
                            if needed and not dp[t][l]:
                                dependency_finished = True
                                for a in self.data.parameters.A[job_idx][operation_idx]:
                                    if (
                                        finish_time[job_idx][a] == -1
                                        or t <= finish_time[job_idx][a]
                                    ):
                                        dependency_finished = False
                                if dependency_finished:
                                    needed -= 1
                                    usage_queue_if_available.append(l)
                        if not needed:
                            x[job_idx][operation_idx] = t
                            for t_prime in range(d):
                                for l in usage_queue_if_available:
                                    dp[t + t_prime][l] = 1
                                y[job_idx][operation_idx][t + t_prime] = 1
                            finish_time[job_idx][operation_idx] = t + d - 1

        obj_from_greedy = self.__cal_obj_numpy(x)

        if os.path.exists(os.path.join("sol", "{}.npy".format(self.m.ModelName))):

            x_saved, y_saved = self.__get_sol_result_params(
                os.path.join("sol", "{}.npy".format(self.m.ModelName))
            )

            if x_saved is not None:
                obj_from_saved = self.__cal_obj_numpy(x_saved)
            else:
                obj_from_saved = np.inf

            print("saved: {}, greedy: {}".format(obj_from_saved, obj_from_greedy))

            if obj_from_saved < obj_from_greedy:
                print("Using saved...")
                x = x_saved
                y = y_saved
            else:
                print("Using greedy...")

        else:
            print("Using greedy...")
            print("Objective value = {}".format(obj_from_greedy))

        # 操作必須連續執行
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                assert np.sum(
                    [y[n][m][i] * y[n][m][i + 1] for i in range(self.T_cnt - 1)]
                ) == max(0, self.data.parameters.D[n][m] - 1)

        # 操作時長等於所需時長
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                assert (
                    np.sum([y[n][m][i] for i in range(self.T_cnt)])
                    == self.data.parameters.D[n][m]
                )

        # Dependency 限制
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                for a in self.data.parameters.A[n][m]:
                    dependency = (
                        np.sum([y[n][a][t] * t for t in range(self.T_cnt)])
                        * 2
                        // self.data.parameters.D[n][a]
                        + 1
                        - self.data.parameters.D[n][a]
                    ) // 2
                    target = (
                        np.sum([y[n][m][t] * t for t in range(self.T_cnt)])
                        * 2
                        // self.data.parameters.D[n][m]
                        + self.data.parameters.D[n][m]
                        - 1
                    ) // 2
                    assert target > dependency

        self.T_cnt = min(
            self.T_cnt,
            int(np.max(np.where(np.sum(np.sum(y, axis=0), axis=0) != 0))) + 5,
        )

        self.output.variables.C_max = self.m.addVar(
            vtype=GRB.INTEGER, lb=0.0, ub=float(self.T_cnt)
        )
        self.output.variables.C = self.m.addVars(
            self.N_cnt, vtype=GRB.INTEGER, lb=0.0, ub=float(self.T_cnt)
        )
        self.output.variables.x = self.m.addVars(
            self.N_cnt, self.M_cnt, vtype=GRB.INTEGER, lb=0.0, ub=float(self.T_cnt)
        )
        self.output.variables.y = self.m.addVars(
            self.N_cnt, self.M_cnt, self.T_cnt, vtype=GRB.BINARY, lb=0.0, ub=1.0
        )

        for i, c in enumerate(np.max(x + self.data.parameters.D - 1, axis=1)):
            self.output.variables.C[i].start = c
        self.output.variables.C_max.start = np.max(
            np.max(x + self.data.parameters.D - 1, axis=1)
        )
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                self.output.variables.x[n, m].start = x[n][m]
                for t in range(self.T_cnt):
                    try:
                        self.output.variables.y[n, m, t].start = y[n][m][t]
                    except:
                        print(n, m, t, self.T_cnt)
                        exit()

    def formulation(self):
        """
        Set Objective
        """
        self.m.setObjective(
            (
                self.output.variables.C_max
                + gp.quicksum(
                    (self.data.parameters.W[n] * self.output.variables.C[n])
                    for n in range(self.N_cnt)
                )
            )
            * self.window,
            GRB.MINIMIZE,
        )
        print("Objective")
        """
        Add Constraints
        """
        # Makespan 的計算
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    for t in range(self.T_cnt):
                        self.m.addConstr(
                            self.output.variables.C[n]
                            >= self.output.variables.x[n, m]
                            + self.data.parameters.D[n][m]
                            - 1
                        )
        for n in range(self.N_cnt):
            self.m.addConstr(self.output.variables.C_max >= self.output.variables.C[n])
        print("c1: Makespan 的計算")
        # 同時間資源使用不能超過總資源數
        for t in range(self.T_cnt):
            self.m.addConstr(
                gp.quicksum(
                    self.data.parameters.S[n][m] * self.output.variables.y[n, m, t]
                    for n in range(self.N_cnt)
                    for m in range(self.M_cnt)
                    if self.data.parameters.S[n][m]
                )
                <= self.L_cnt
            )
        print("c2: 同時間資源使用不能超過總資源數")
        # 操作必須連續執行
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    self.m.addConstr(
                        (
                            gp.quicksum(
                                self.output.variables.y[n, m, t] * t
                                for t in range(self.T_cnt)
                            )
                            * 2
                            / self.data.parameters.D[n][m]
                            + 1
                            - self.data.parameters.D[n][m]
                        )
                        / 2
                        >= self.output.variables.x[n, m]
                    )
                    for t in range(self.T_cnt):
                        self.m.addConstr(
                            self.output.variables.y[n, m, t] * t
                            <= self.output.variables.x[n, m]
                            + self.data.parameters.D[n][m]
                            - 1
                        )
        print("c3: 操作必須連續執行")
        # 操作時長等於所需時長
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                if self.data.parameters.S[n][m]:
                    self.m.addConstr(
                        gp.quicksum(
                            self.output.variables.y[n, m, t] for t in range(self.T_cnt)
                        )
                        == self.data.parameters.D[n][m]
                    )
        print("c4: 操作時長等於所需時長")
        # Dependency 限制
        for n in range(self.N_cnt):
            for m in range(self.M_cnt):
                for m_prime in self.data.parameters.A[n][m]:
                    self.m.addConstr(
                        self.output.variables.x[n, m]
                        >= self.output.variables.x[n, m_prime]
                        + self.data.parameters.D[n][m_prime]
                    )
        print("c5: Dependency 限制")

    def optimize(self, time_limit=np.inf, target=-np.inf):
        """
        Optimization
        """
        self.m.setParam("Timelimit", time_limit)
        self.m.params.BestObjStop = target
        self.m.optimize(callback=save_checkpoint)