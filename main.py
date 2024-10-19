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


def main():
    interior_point_algorithm(
        initial_x=[1, 1, 1, 315, 174, 169],
        A=[
            [18, 15, 12, 1, 0, 0],
            [6, 4, 8, 0, 1, 0],
            [5, 3, 3, 0, 0, 1]
        ],
        c=[9, 10, 16, 0, 0, 0],
        alpha=0.5,
        epsilon=0.0001,
    )


if __name__ == "__main__":
    main()
