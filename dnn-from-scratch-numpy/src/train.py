from src.data import make_data
from src.model import *
from src.utils import *

def train(layer_dims, activation, learning_rate, num_iters):
    X, Y = make_data()

    best_cost = np.inf
    costs = []
    parameters = initialize_parameters(layer_dims, seed=42)
    best_parameters = {k: v.copy() for k, v in parameters.items()}

    for i in range(num_iters):

        caches, A_L = L_model_forward(X, parameters, activation)

        cost = cost_function(A_L, Y)

        costs.append(cost)
        if cost < best_cost:
            best_cost = cost

            best_parameters = {k: v.copy() for k, v in parameters.items()}

        grads = L_model_backward(A_L, Y, caches, activation)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(f"Iteration {i}, Cost = {cost:.4f}")
        
    _, A_L_best = L_model_forward(X, best_parameters, activation)
    Y_pred = (A_L_best >= 0.5).astype(int)
    accuracy = np.mean(Y_pred == Y)
    print("Final Accuracy (best params):", accuracy)

    return best_parameters, best_cost, costs, A_L_best, Y


