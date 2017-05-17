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

    model.fit(X[:, 0:1], y, X)

    fig, ax = plt.subplots()

    ax.plot(X[y==1, 0], X[y==1,1], 'bo',
            X[y==-1, 0], X[y==-1, 1], 'ro')

if __name__ == "__main__":
    main()
