import numpy as np

def initialize_parameters(layer_dims, seed = 42):

    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters


def L_model_forward(X, parameters, activation):
    A = X
    L = len(parameters)//2
    caches = []

    for l in range(1, L + 1):
        A_prev = A
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        Z = np.dot(W, A) + b
        
        act = activation[l-1]
        if act == "relu":
            A = np.maximum(0, Z)
        elif act == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("unknown activation")
        cache = (A_prev, W, b, Z, A)
        caches.append(cache)
    A_L = A
    return caches, A_L 


def cost_function(A_L, Y):
    m = Y.shape[1]
    A_L = np.clip(A_L, 1e-10, 1-1e-10)
    cost = - (1/m) * np.sum(
        Y * np.log(A_L) +
        (1 - Y) * np.log(1 - A_L)
    )
    return float(np.squeeze(cost))


def L_model_backward(A_L, Y, caches, activation):
    grads = {}
    L = len(caches)
    m = Y.shape[1]

    last_cache = caches[-1]
    A_prev, W, b, Z, A = last_cache
    
    dZ = A_L - Y

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA = np.dot(W.T, dZ)
    
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db


    for l in reversed(range(L-1)):
        cache = caches[l]
        A_prev, W, b, Z, A = cache

        act = activation[l]
        
        if act == "relu":
            dZ = dA * (Z > 0).astype(float)

        elif act == "sigmoid":
            dZ = dA * A * (1 - A)
            
        else:
            raise ValueError("unknown activation in backward")

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(W.T, dZ)
        
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db

    return grads
        

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return parameters