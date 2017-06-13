import datetime
import numpy as np
import pandas as pd
import data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from progressbar import ProgressBar

try:
    from gp import privEPClassification, kernel
    from gp import EPClassification
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import privEPClassification, kernel
    from gp import EPClassification


def test0():
    """test with synthesis twin moon data"""
    X, y = data.twin(200)
    model = privEPClassification()
    model.fit(X[:,1:], y, X)

    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)

    print(model.log_ml())
    #model.empiricalBayes()
    #print(model.log_ml())
    decfun = model.decision_function(Xtest[:,1:])
    pred = model.predict(Xtest[:,1:])

    fig, ax = plt.subplots()
    #cs = ax.contourf(Xmesh[0], Xmesh[1], pred.reshape(*Xmesh[0].shape))
    cs = ax.contourf(Xmesh[0], Xmesh[1], decfun.reshape(*Xmesh[0].shape))
    fig.colorbar(cs)
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo',
            X[y == -1, 0], X[y == -1, 1], 'ro')
    plt.show()
    plt.savefig('output.png')

def test1():
    X, y = data.australian()

    result = []
    for sigmax, betax, sigmaz, betaz in [(sx, bx, sz, bz)
            for sx in 2**np.arange(-5.0, 5.0, 2.0)
            for bx in 2**np.arange(-5.0, 5.0, 2.0)
            for sz in 2**np.arange(-5.0, 5.0, 2.0)
            for bz in 2**np.arange(-5.0, 5.0, 2.0)]:
        for xdim in range(2, X.shape[1]):
            priv_accuracy = []
            for _ in range(5):
                idx = np.random.permutation(len(y))
                Xtr = X[idx[:300]]
                ytr = y[idx[:300]]
                Xte = X[idx[300:]]
                yte = y[idx[300:]]

                model = privEPClassification(kernel=kernel.PrivMultRBF,
                        sigmax=sigmax, betax=betax,
                        sigmaz=sigmaz, betaz=betaz)
                model.fit(Xtr[:, :xdim], ytr, Xtr)
                #model.empiricalBayes()
                
                ypr = model.predict(Xtr[:,:xdim])
                train_acc = accuracy_score(ytr, ypr)
                ypr = model.predict(Xte[:,:xdim])
                test_acc = accuracy_score(yte, ypr)
                priv_accuracy.append(test_acc)

                result.append({'sigmax':sigmax, 'betax':betax, 'xdim':xdim,
                    'sigmaz':sigmaz, 'betaz':betaz,
                    'test_accuracy':test_acc,
                    'train_accuracy':train_acc})

    pd.DataFrame(result).to_csv('priv_result')

def test2():
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    X, y = data.australian()
    ntrain = 300
    params = {'sigma': 2**np.arange(-3.0, 6.0, 2.0),
              'beta': 2**np.arange(-3.0, 6.0, 2.0)}
    xdims = range(2, X.shape[1])
    alphas = np.arange(0, 1.2, 0.2)
    cv = 5
    repeat = 5
    result = []

    with open('meta%s.txt'%time, 'w') as f:
        print("Classification with privileged information by EPGPC", file=f)
        print("Australian data", file=f)
        print("X.shape =", X.shape, file=f)
        print("ntrain =", ntrain, file=f)
        print("params =", params, file=f)
        print("cv =", cv, file=f)
        print("xdims =", xdims, file=f)
        print("alphas =", alphas, file=f)
        print("repeat =", repeat, file=f)

    p = ProgressBar(repeat*len(xdims)*len(alphas))
    i = 1

    for _ in range(repeat):
        p.update(i)
        i += 1
        idx = np.random.permutation(len(y))
        Xtr = X[idx[:ntrain]]
        ytr = y[idx[:ntrain]]
        Xte = X[idx[ntrain:]]
        yte = y[idx[ntrain:]]

        model = GridSearchCV(EPClassification(), params, cv=cv, n_jobs=-1)
        model.fit(Xtr, ytr)
        sigmaz = model.best_params_['sigma']
        betaz = model.best_params_['beta']

        for xdim in xdims:
            model = GridSearchCV(EPClassification(), params, cv=cv, n_jobs=-1)
            model.fit(Xtr[:, :xdim], ytr)
            sigmax = model.best_params_['sigma']
            betax = model.best_params_['beta']

            for alpha in alphas:
                model = privEPClassification(alpha=alpha,
                        sigmax=sigmax, betax=betax, sigmaz=sigmaz, betaz=betaz)
                model.fit(Xtr[:, :xdim], ytr, Xtr)
                accuracy = model.score(Xte[:, :xdim], yte)

                result.append({'xdim':xdim, 'alpha':alpha, 'accuracy':accuracy,
                    'sigmax':sigmax, 'betax':betax, 'sigmaz':sigmaz, 'betaz':betaz})

    pd.DataFrame(result).to_csv('result%s.csv'%time)


def test3():
    assert False
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

def debug0():
    assert False
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

def debug():
    from gp.kernel import PrivMultRBF
    X = np.random.rand(10, 3)
    Z = np.random.rand(10, 10)
    K1 = PrivMultRBF((X, Z), (X, Z))()
    K12 = PrivMultRBF((X, Z), (X, None), Zsample=Z)()
    K2 = PrivMultRBF((X, None), (X, None), Zsample=Z)()
    K = np.concatenate([
        np.concatenate([K1, K12], axis=1),
        np.concatenate([K12.T, K2], axis=1)])

    print(np.linalg.eigvals(K))

if __name__ == "__main__":
    test2()
else:
    X, y = data.twin(10)
    model = EPClassification(X, y)
    Xmesh = np.meshgrid(np.linspace(-2, 2), np.linspace(-1.5, 1.5))
    Xtest = np.stack(Xmesh, 2).reshape(-1, 2)
    decfun = model.decision_function(Xtest)
