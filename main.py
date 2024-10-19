import numpy as np
from numpy.linalg import norm


def is_inapplicable(
        A: list[list[float]],
        c: list[float],
) -> bool:
    is_inapplicable_flag = False
    for i in range(len(c)):
        if -c[i] <= 0:
            # print(c[i])
            is_inapplicable_flag = True
            for j in range(len(A)):
                if A[j][i] > 0: is_inapplicable_flag = False
            if is_inapplicable_flag: break
    return is_inapplicable_flag


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

    iteration = 1

    while True:

        if is_inapplicable(initial_A, initial_c):
            print("Method is not applicable")
            break

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
            print("In the last iteration ", iteration, "we have x =\n", x, "with alpha = ", alpha)
            print("Value of objective function is: ", np.dot(c, np.transpose(x)))
            break


def read_input():
    vector_x = [float(i) for i in input(
        "Print the initial trial solution that lies in the interior of the feasible region, i.e. inside the boundary of the feasible region: ", ).split()]
    n = int(input("Print the number of constraints functions: ", ))
    print("Input constraints functions line by line:")
    matrix_A = []

    for _ in range(n):
        a_i = [float(i) for i in input().split()]
        matrix_A.append(a_i)

    vector_c = [float(i) for i in input(
        "A vector of coefficients of objective function:", ).split()]

    return vector_x, matrix_A, vector_c


def main():
    vector_x, matrix_A, vector_c = read_input()

    interior_point_algorithm(
        initial_x=vector_x,
        initial_A=matrix_A,
        initial_c=vector_c,
        alpha=0.5,
        epsilon=0.00001,
    )

    interior_point_algorithm(
        initial_x=vector_x,
        initial_A=matrix_A,
        initial_c=vector_c,
        alpha=0.9,
        epsilon=0.00001,
    )


if __name__ == "__main__":
    main()
