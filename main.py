import os
import sys
from data_loader import ModelData, ModelVars
from model import Model
from formatting import gen_output_to_judge, shift_result


def checkAns(in_, out_):
    os.system(
        "{} --public {} {}".format(
            os.path.join("ada-final-public", "checker"),
            os.path.join("ada-final-public", in_),
            out_,
        )
    )


if __name__ == "__main__":

    assert len(sys.argv) == 2

    idx = int(sys.argv[1])

    tuning_windows = [1, 1, 1, 400, 800, 20, 60, 100, 10, 40, 20]
    tuning_sortby = [1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2]
    public_problem_sets = [
        "00.in",
        "01.in",
        "02.in",
        "03.in",
        "04.in",
        "05.in",
        "06.in",
        "07.in",
        "08.in",
        "09.in",
        "10.in",
    ]
    best_stop_objs = [
        0,
        67778.8,
        174,
        118417.256,
        281538.264,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    problem = public_problem_sets[idx]
    problem_prefix = problem.split(".")[0]
    output_path = os.path.join("ans", "{}.out".format(problem_prefix))

    model_input = ModelData(os.path.join("ada-final-public", problem))
    model_output = ModelVars()

    model = Model(problem_prefix + ".sol", model_input, model_output, problem)
    model.pre_solve(window=tuning_windows[idx], sort_num=tuning_sortby[idx])
    model.formulation()
    model.optimize(target=best_stop_objs[idx])

    gen_output_to_judge(
        model.gen_operations_order(problem_prefix),
        model.L_cnt,
        model.T_cnt,
        output_path,
    )

    shift_result(
        output_path,
        output_path,
        len(model_input.sets.L),
        len(model_input.sets.N),
        model_input.sets.M,
        model_input.parameters.D,
        model_input.parameters.A,
    )

    checkAns(problem, output_path)
