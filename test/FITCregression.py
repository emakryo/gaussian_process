import matplotlib.pyplot as plt
import data

try:
    from gp import FITCRegression
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import FITCRegression


def main():
    X, y = data.data1(10000, 0.2)
    model = FITCRegression(10)
    model.fit(X, y)
    # for i in range(10):
    #     model.optimize(True)

    model.plot()
    print(model.logMarginal())
    plt.show()
    # print(model.grad_log_marginal_likelihood())
    # print(model.log_marginal_likelihood())
    # print(model.param_array)
    # print(model.k_params)


if __name__ == "__main__":
    main()
