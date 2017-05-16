import matplotlib.pyplot as plt
import data

try:
    from gp import Regression
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import Regression


def main():
    X, y = data.data1(100, 0.2)
    model = Regression()
    model.fit(X, y)
    # for i in range(10):
    #     model.optimize(True)

    model.plot("output.png")
    print(model.log_marginal_likelihood())
    plt.show()
    # print(model.grad_log_marginal_likelihood())
    # print(model.param_array)
    # print(model.k_params)

#    x = np.linspace(0.002, 0.01, 10)
#    y = np.zeros(10)
#    yg = np.zeros(10)
#    for i,xn in enumerate(x):
#        param_array = model.param_array
#        param_array[0] = xn
#        model.param_array = param_array
#        y[i]  = model.log_marginal_likelihood()
#        yg[i] = model.grad_log_marginal_likelihood()[0]
#
#    plt.clf()
#    plt.plot(x,y)
#    plt.savefig('obj.png')
#    plt.clf()
#    plt.plot(x,yg)
#    plt.savefig('grad.png')


if __name__ == "__main__":
    main()
