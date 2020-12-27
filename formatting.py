import numpy as np


def gen_output_to_judge(results, N, T, path):
    dp = np.zeros((T, N)).astype("int")
    results = sorted(results, key=lambda x: x[2])
    for idx, result in enumerate(results):
        needed = result[3]
        for i, used in enumerate(dp[result[2]]):
            if needed and not used:
                dp[result[2]][i] = 1
                results[idx][5].append(i + 1)
                needed -= 1
                for j in range(result[2], result[2] + result[4], 1):
                    dp[j][i] = 1
    results = sorted(results, key=lambda x: (x[0], x[1]))
    with open(path, "w") as f:
        for result in results:
            f.write("{} {}\n".format(result[2], " ".join([str(r) for r in result[5]])))


def shift_result(path, output, L, N, M, D, A):

    with open(path, "r") as f:
        simplified_results = [
            [int(y) for y in x.split()] for x in f.read().strip().split("\n")
        ]

    idx = 0
    results = []
    for n in range(N):
        res = []
        for m in range(M[n]):
            res.append([(n, m), simplified_results[idx]])
            idx += 1
        results.append(res)

    simplified_results = []
    for n in range(N):
        for m in range(M[n]):
            simplified_results.append(results[n][m])

    simplified_results = sorted(simplified_results, key=lambda x: x[1][0])

    cur_time = np.zeros(L).astype("int")
    for idx, result in enumerate(simplified_results):

        tmp_max = 0
        for resource in result[1][1:]:
            tmp_max = max(tmp_max, cur_time[resource - 1])
        for a in A[result[0][0]][result[0][1]]:
            for res in simplified_results:
                if result[0][0] == res[0][0] and a == res[0][1]:
                    tmp_max = max(tmp_max, res[1][0] + D[res[0][0]][res[0][1]])

        result[1][0] = tmp_max

        for resource in result[1][1:]:
            cur_time[resource - 1] = tmp_max + D[result[0][0]][result[0][1]]

    simplified_results = sorted(simplified_results, key=lambda x: x[0])

    with open(output, "w") as f:
        for results in simplified_results:
            f.write(" ".join([str(x) for x in results[1]]))
            f.write("\n")
