import numpy as np
import matplotlib.pyplot as plt
import data

try:
    from gp import PrivClassification
except:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from gp import PrivClassification

def main():
    X, y = data.twin(50)
    model = PrivClassification(50)

    model.fit(X[:, 1:2], y, X)
    
    Xtest = np.linspace(-5, 5).reshape(-1, 1)
    decfun = model.predict(Xtest)
    decfun = np.tile(decfun.reshape(-1, 1), (1, 50))

    fig, ax = plt.subplots()

    ax.plot(X[y==1, 0], X[y==1,1], 'bo',
            X[y==-1, 0], X[y==-1, 1], 'ro')

    ax.contourf(np.linspace(-5, 5), np.linspace(-5, 5), decfun)

    plt.show()

if __name__ == "__main__":
    main()
