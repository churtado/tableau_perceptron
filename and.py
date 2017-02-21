import tabpy_client
from sklearn.linear_model import Perceptron

# create a perceptron
ppn = Perceptron(n_iter=10, eta0=0.1, random_state=0)

# declare the training set
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]

# train the perceptron with the training set
ppn.fit(X, y)

def predict(x, y):
    import numpy as np
    X = np.column_stack(((x,y)))
    return ppn.predict(X).tolist()


client = tabpy_client.Client('http://localhost:9004/')
client.deploy('predict', predict, 'Predicts the AND of 2 numbers using a perceptron', override=True)

x = [0,0,1,1]
y = [0,1,0,1]

print(client.query('predict', x, y))
