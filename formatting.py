import numpy as np


def gen_output_to_judge(results, N, T, path):
    dp = np.zeros((T, N)).astype('int')
    results = sorted(results, key=lambda x: x[2])
    for idx, result in enumerate(results):
        needed = result[3]
        for i, used in enumerate(dp[result[2]]):
            if needed and not used:
                dp[result[2]][i] = 1
                results[idx][5].append(i+1)
                needed -= 1
                for j in range(result[2], result[2]+result[4], 1):
                    dp[j][i] = 1
    results = sorted(results, key=lambda x: (x[0], x[1]))
    with open(path, 'w') as f:
        for result in results:
            f.write('{} {}\n'.format(result[2], ' '.join([str(r) for r in result[5]])))