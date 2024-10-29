from json.encoder import INFINITY
from decimal import *


# Prints the basic variables as "x1 = basic_variables_values[0], x2 = basic_variables_values[1],... "
def format_output(basic_variables_values):
    formatted_output = ", ".join(f"x{i + 1} = {value}" for i, value in enumerate(basic_variables_values))
    print(formatted_output)


class Simplex:
    def __init__(self, z, constraints, constraints_rhs, accuracy):
        self._z = z
        self._constraints = constraints
        self._constraints_rhs = constraints_rhs
        self._solution = 0
        self._num_of_const = len(constraints)
        self._basic_variables_number = len(constraints)
        self._variables_number = self._basic_variables_number + self._num_of_const
        self._basic_variables = list(range(self._basic_variables_number))
        self._basis = list(
            range(self._basic_variables_number,
                  self._basic_variables_number + self._num_of_const)
        )
        self._accuracy = accuracy
        self.__equally_negative_variables = []

        for i in range(self._num_of_const):
            constraints[i] += [0] * self._num_of_const
            constraints[i][len(self._z) + i] = 1

        for i in range(len(z)):
            self._z[i] *= -1

        self._z += [0] * self._num_of_const

    # Computes the maximum value for z
    def compute_maximum(self):
        while True:
            if not self._can_continue():
                basic_variables_values = self._get_basic_variables_values()
                self._final_format_variables(basic_variables_values)
                format_output(basic_variables_values)
                self._solution = self._final_format_solution(self._solution)
                return self._solution

            entering = self._define_entering()

            # Checking if at least one of the possible entering values is applicable
            for variable in self.__equally_negative_variables:
                if self._is_applicable(variable):
                    entering = variable
                    break

            if not self._is_applicable(entering):
                print("The Simplex method is not applicable!")
                return None

            leaving = self._define_leaving(entering)
            self._change_pivot(entering, leaving)

    # Computes the minimum value for z
    def compute_minimum(self):
        self._z = [-variable for variable in self._z]
        return -self.compute_maximum()

    # Gets the index of the entering variable from the z-row
    def _define_entering(self):
        min_z = 0
        entering_index = -1

        for variable in self._z:
            if variable >= 0:
                continue

            if variable < min_z:
                min_z = variable
                entering_index = self._z.index(variable)

        return entering_index

    # Gets the index of the leaving variable from the z-row
    def _define_leaving(self, entering_index):
        min_elem = INFINITY
        basis_leaving_index = -1

        for i in range(self._num_of_const):
            if self._constraints[i][entering_index] <= 0:
                continue

            ratio = self._constraints_rhs[i] / self._constraints[i][entering_index]

            if ratio < min_elem:
                min_elem = ratio
                basis_leaving_index = i

        leaving_index = self._basis[basis_leaving_index]

        return leaving_index

    # Changes the "table" for a new pivot
    def _change_pivot(self, entering_index, leaving_index):
        basis_index = self._basis.index(leaving_index)
        self._basis[basis_index] = entering_index

        self._update_constraints(basis_index, entering_index)
        self._update_z_row(basis_index, entering_index)

    # Updates the whole "table"
    def _update_constraints(self, basis_index, entering_index):
        for constr_id in range(self._num_of_const):
            if constr_id == basis_index:
                self._normalize_pivot_row(constr_id, entering_index)
            else:
                self._update_non_pivot_row(constr_id, basis_index, entering_index)

    # Normalizes the pivot row by dividing all elements by the pivot value
    def _normalize_pivot_row(self, constr_id, entering_index):
        divisor = self._constraints[constr_id][entering_index]
        for var_id in range(self._variables_number - 1):
            self._constraints[constr_id][var_id] = self._constraints[constr_id][var_id] / divisor

        self._constraints_rhs[constr_id] = self._constraints_rhs[constr_id] / divisor

    # Updates values in the non-pivot row
    def _update_non_pivot_row(self, constr_id, basis_index, entering_index):
        factor = -(self._constraints[constr_id][entering_index] /
                   self._constraints[basis_index][entering_index])
        for var_id in range(self._variables_number - 1):
            self._constraints[constr_id][var_id] = self._constraints[constr_id][var_id] + \
                                                   factor * self._constraints[basis_index][var_id]
        self._constraints_rhs[constr_id] = self._constraints_rhs[constr_id] + \
                                           factor * self._constraints_rhs[basis_index]

    # Updates values in the z-row
    def _update_z_row(self, basis_index, entering_index):
        factor_z = -(self._z[entering_index] /
                     self._constraints[basis_index][entering_index])
        for var_id in range(self._variables_number - 1):
            self._z[var_id] = self._z[var_id] + \
                              factor_z * self._constraints[basis_index][var_id]
        self._solution = self._solution + \
                         factor_z * self._constraints_rhs[basis_index]

    # Gets the basic variables values from the constraints_rhs "column"
    def _get_basic_variables_values(self):
        basic_variables_values = []

        for basic_id in self._basic_variables:
            if basic_id in self._basis:
                index = self._basis.index(basic_id)
                basic_variables_values.append(self._constraints_rhs[index])
            else:
                basic_variables_values.append(0)

        return basic_variables_values

    # Returns True if there is at least one negative element in z-row
    def _can_continue(self):
        for variable in self._z:
            if variable < 0: return True
        return False

    # Returns True if for entering variable
    # there is at least one positive value in constraints
    def _is_applicable(self, entering_index):
        for i in range(self._num_of_const):
            if self._constraints[i][entering_index] > 0: return True
        return False

    def _final_format_variables(self, basic_variables_values):
        quant = Decimal("1." + "0" * (self._accuracy))
        for i, num in enumerate(basic_variables_values):
            string_num = Decimal(str(num))
            if 'E' in str(string_num.quantize(quant)):
                basic_variables_values[i] = "0." + "0" * (self._accuracy)
            else:
                basic_variables_values[i] = str(string_num.quantize(quant))

    def _final_format_solution(self, solution):
        quant = Decimal("1." + "0" * self._accuracy)
        string_num = Decimal(str(solution))
        if 'E' in str(string_num.quantize(quant)):
            solution = "0." + "0" * (self._accuracy + 1)
        else:
            solution = string_num.quantize(quant)
        return str(solution)[:-1]


# Reads input and gets z, constraints, constraints_rhs, and accuracy
def read_input():
    z = list(map(float, input("A vector of coefficients of objective function: ").split()))
    num_of_const = int(input("Number of constraints: "))
    print("A matrix of coefficients of constraint function:")
    constraints = []
    constraint = list(map(float, input().split()))
    constraints.append(constraint)

    for _ in range(num_of_const - 1):
        constraint = list(map(float, input().split()))
        constraints.append(constraint)

    constraints_rhs = list(map(float, input("A vector of right-hand side numbers: ").split()))

    accuracy = int(input("The approximation accuracy: "))

    return z, constraints, constraints_rhs, accuracy


def main():
    z, constraints, constraints_rhs, accuracy = read_input()
    simplex_method = Simplex(z, constraints, constraints_rhs, accuracy)
    maximum = simplex_method.compute_maximum()
    if not maximum is None: print(maximum)


if __name__ == '__main__':
    main()
