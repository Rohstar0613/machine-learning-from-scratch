from src.train import train
from src.utils import *

def main():
    layer_dims = [54, 32, 16, 1]
    activation = ["relu", "relu",  "sigmoid"]
    learning_rate = 0.1
    num_iters = 10000

    best_parameters, best_cost, costs, A_L_best, Y = train(layer_dims, activation, learning_rate, num_iters)

    save_parameters(best_parameters, filename="data/best_params.npz")

    print_bestcost(best_cost)

    plot_learning_curve(costs)

    evaluate_model(A_L_best, Y)


if __name__ == "__main__":

    main()
