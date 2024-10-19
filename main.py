import numpy as np
from numpy.linalg import norm


def interior_point_algorithm(
        initial_x: list[int],
        A: list[list[int]],
        c: list[int],
        alpha: float,
        epsilon: float,
):
    """
    Performs the interior point algorithm for optimization.

    :param initial_x: Initial value of the variable x.
    :param A: Coefficient matrix.
    :param c: Coefficients of the objective function.
    :param alpha: Parameter controlling the step size.
    :param epsilon: Convergence parameter.
    """
    x = np.array(initial_x, float)
    A = np.array(A, float)
    c = np.array(c, float)

    iteration = 1

    while True:
        v = x
        D = np.diag(x)
        AA = np.dot(A, D)
        cc = np.dot(D, c)
        I = np.eye(len(c))
        F = np.dot(AA, np.transpose(AA))
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(I, np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.abs(np.min(cp))
        y = np.add(np.ones(len(c), float), (alpha / nu) * cp)
        yy = np.dot(D, y)
        x = yy

        iteration += 1

        if norm(np.subtract(yy, v), ord=2) < epsilon:
            break

    print("In the last iteration", iteration, "we have x =\n", *x)


def read_input():
    vector_x = [int(i) for i in input(
        "Print the initial trial solution that lies in the interior of the feasible region, i.e. inside the boundary of the feasible region: ", ).split()]
    n = int(input("Print the number of constraints functions: ", ))
    print("Input constraints functions line by line:")
    matrix_A = []

    for _ in range(n):
        a_i = [int(i) for i in input().split()]
        matrix_A.append(a_i)

    vector_c = [int(i) for i in input(
        "A vector of coefficients of objective function:", ).split()]

    return vector_x, matrix_A, vector_c


def main():
    vector_x, matrix_A, vector_c = read_input()

    interior_point_algorithm(
        initial_x=vector_x,
        A=matrix_A,
        c=vector_c,
        alpha=0.5,
        epsilon=0.0001,
    )


if __name__ == "__main__":
    main()
