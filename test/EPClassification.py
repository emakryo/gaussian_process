import numpy as np
import data
import matplotlib.pyplot as plt

try:
    from gp import EPClassification
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import EPClassification


def main():
    X, y = data.twin(200)
    model = EPClassification()
    model.fit(X, y)

    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)
    decfun = model.decision_function(Xtest)

    fig, ax = plt.subplots()
    cs = ax.contourf(Xmesh[0], Xmesh[1], decfun.reshape(*Xmesh[0].shape))
    fig.colorbar(cs)
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo',
            X[y == -1, 0], X[y == -1, 1], 'ro')
    plt.show()

if __name__ == "__main__":
    main()
else:
    X, y = data.twin(10)
    model = EPClassification(X, y)
    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)
    decfun = model.decision_function(Xtest)
