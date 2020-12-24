import os
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

    problem = "05.in"

    model_input = ModelData(os.path.join("ada-final-public", problem))
    model_output = ModelVars()

    shift_result(
        "./ans/05.out",
        len(model_input.sets.L),
        len(model_input.sets.N),
        model_input.sets.M,
        model_input.parameters.D,
        model_input.parameters.A,
    )
    exit()

    model = Model("ada_final", model_input, model_output, problem)
    results = model.pre_solve(window=40)
    model.formulation()
    model.optimize(time_limit=14000)
    results = model.dump_results()

    gen_output_to_judge(
        results,
        model.L_cnt,
        model.T_cnt,
        os.path.join("ans", "{}.out".format(problem.split(".")[0])),
    )

    checkAns(problem, os.path.join("ans", "{}.out".format(problem.split(".")[0])))
