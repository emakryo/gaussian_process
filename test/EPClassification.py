import numpy as np
import pandas as pd
import data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

try:
    from gp import EPClassification
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import EPClassification


def test0():
    """test with synthesis twin moon data"""
    X, y = data.twin(200)
    model = EPClassification(sigma=1.0, beta=10)
    model.fit(X, y)

    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)

    print(model.log_ml())
    #model.empiricalBayes()
    print(model.log_ml())
    decfun = model.decision_function(Xtest)
    pred = model.predict(Xtest)

    fig, ax = plt.subplots()
    cs = ax.contourf(Xmesh[0], Xmesh[1], pred.reshape(*Xmesh[0].shape))
    #cs = ax.contourf(Xmesh[0], Xmesh[1], decfun.reshape(*Xmesh[0].shape))
    fig.colorbar(cs)
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo',
            X[y == -1, 0], X[y == -1, 1], 'ro')
    plt.show()
    plt.savefig('output.png')

def test1():
    X, y = data.australian()

    result = []
    for sigma, beta in [(s, b) for s in 2**np.arange(-5.0, 5.0)
                        for b in 2**np.arange(-5.0, 5.0)]:
        accuracy = []
        for _ in range(5):
            idx = np.random.permutation(len(y))
            Xtr = X[idx[:300]]
            ytr = y[idx[:300]]
            Xte = X[idx[300:]]
            yte = y[idx[300:]]
            model = EPClassification(sigma=sigma, beta=beta)
            model.fit(Xtr, ytr)
            #model.empiricalBayes()
            
            train_acc = accuracy_score(ytr, model.predict(Xtr))
            ypr = model.predict(Xte)
            test_acc = accuracy_score(yte, ypr)
            accuracy.append(test_acc)

            result.append({'sigma':sigma, 'beta':beta, 'test_accuracy':test_acc,
                'train_accuracy':train_acc})

        print(sigma, beta, np.mean(accuracy), np.std(accuracy))

    pd.DataFrame(result).to_csv('result')

def test2():
    X, y = data.australian()

    accuracy = []
    for _ in range(5):
        idx = np.random.permutation(len(y))
        Xtr = X[idx[:300]]
        ytr = y[idx[:300]]
        Xte = X[idx[300:]]
        yte = y[idx[300:]]
        model = SVC()
        model.fit(Xtr, ytr)
        
        ypr = model.predict(Xte)
        accuracy.append(accuracy_score(yte, ypr))

    print(np.mean(accuracy), np.std(accuracy))

def debug():
    X, y = data.twin(50)
    model = EPClassification()

    surface = np.empty((50, 50, 11))
    quiver = np.empty((50, 50, 2))
    sigma = np.linspace(1, 20)
    beta = np.linspace(1, 20)
    for i, j in [(i, j) for i in range(50) for j in range(50)]:
        model = EPClassification(k_params={'sigma':sigma[i], 'beta':beta[j]})
        model.fit(X, y)
        surface[i][j] = model.log_ml_terms()
        quiver[i][j] = model.dlog_mldtheta()

    np.save("marginal", surface)
    np.save("grad_marginal", quiver)

if __name__ == "__main__":
    test1()
else:
    X, y = data.twin(10)
    model = EPClassification(X, y)
    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)
    decfun = model.decision_function(Xtest)
