import math

import numpy as np
from numpy.linalg import norm

from simplex import Simplex


def interior_point_algorithm(
        initial_x: list[float],
        initial_A: list[list[float]],
        initial_c: list[float],
        alpha: float,
        epsilon: float,
) -> None:
    """
    Performs the interior point algorithm for optimization.

    :param initial_x: Initial value of the variable x.
    :param initial_A: Coefficient matrix.
    :param initial_c: Coefficients of the objective function.
    :param alpha: Parameter controlling the step size.
    :param epsilon: Convergence parameter.
    """
    x = np.array(initial_x, float)
    A = np.array(initial_A, float)
    c = np.array(initial_c, float)

    iteration = 0

    while True:
        iteration += 1
        try:
            v = x
            D = np.diag(x)

            A_tilde = np.dot(A, D)
            c_tilde = np.dot(D, c)

            I = np.eye(len(c))

            A_tilde_A_tr = np.dot(A_tilde, np.transpose(A_tilde))
            A_tilde_A_tr_inverse = np.linalg.inv(A_tilde_A_tr)
            H = np.dot(np.transpose(A_tilde), A_tilde_A_tr_inverse)

            P = np.subtract(I, np.dot(H, A_tilde))

            c_p = np.dot(P, c_tilde)

            if np.min(c_p) >= 0:
                print("Method is not applicable")
                break
            nu = np.abs(np.min(c_p))

            x_tilde = np.add(np.ones(len(c), float), (alpha / nu) * c_p)
            new_x = np.dot(D, x_tilde)
            x = new_x
        except():
            print("Method is not applicable")
            break

        if norm(np.subtract(new_x, v), ord=2) < epsilon:
            print(
                f"In the last iteration {iteration} we have x =", x,
                f"with alpha = {alpha}",
                sep='\n'
            )
            print("Value of objective function is: ", np.dot(c, np.transpose(x)))
            break


def read_input():
    vector_c = [float(c) for c in input(
        "A vector of coefficients of objective function: "
    ).split()]

    n = int(input("The number of constraints functions: "))

    print("Constraints functions line by line: ")
    matrix_A = []

    for _ in range(n):
        a_i = [float(a_i_j) for a_i_j in input().split()]
        matrix_A.append(a_i)

    vector_x = [float(x) for x in input(
        "The initial point: "
    ).split()]

    constraints_rhs = list(map(float, input("A vector of right-hand side numbers: ").split()))

    accuracy = int(input("Approximation accuracy: "))
    epsilon = 0.1 ** accuracy

    return vector_x, matrix_A, vector_c, epsilon, constraints_rhs


def main():
    vector_x, matrix_A, vector_c, epsilon, constraints_rhs = read_input()

    interior_point_algorithm(
        initial_x=vector_x,
        initial_A=matrix_A,
        initial_c=vector_c,
        alpha=0.5,
        epsilon=epsilon,
    )

    interior_point_algorithm(
        initial_x=vector_x,
        initial_A=matrix_A,
        initial_c=vector_c,
        alpha=0.9,
        epsilon=epsilon,
    )

    simplex = Simplex(
        vector_c,
        matrix_A,
        constraints_rhs,
        int(math.log(epsilon, 0.1))
    )
    simplex.compute_maximum()


if __name__ == "__main__":
    main()
