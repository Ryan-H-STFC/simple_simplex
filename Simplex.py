import numpy as np


"""
Setup -

Step 1:
    Turn Inequalities into Equalities with Slack Variables
    A[n x m] n - constraints, m - variables
    Basis Variables are Unit Columns

    Find Objective Coefficients of the basis variables (CB)
    z[j] = dot product of CB with columns of Z
    Net Evalution - CB[j] - z[j]

    Find Largest of net-eval,  effectively most negative z[j] set as Pivot Column  (Entering Variable)
    Calc b-ratio -> b / ratio, if negative set as infinity
    Pivot Row = Minimum Non-Negative b-ratio                                       (Leaving Variable)

    pivot element found

Iterations Start-

Step 3:
    CB Coefficients update using new basis, (Entering Variable Swaps Leaving)
    Step 3a:
        ? # Normalise Pivot Row by Pivot element
        Pivot Row New = Pivot Row / Pivot Element
        b[j] / Pivot Element

        #? Aim: Pivot column elements become zero besides pivot element, for each value in a row subtract the
        #? corresponding value in the pivot column multiplied with the pivot row
        New Rows = Row - pivot[row] * Pivot Row

    Step 3b:
        Calculate new z[j]
        Calculate new Net Evalution CB[j] - z[j]

    Step 3c:
        If All Net Evaluation are negative Stop:
            optimal solution found
        else:
            Repeat step 3

"""


def simplex(
    A: np.ndarray[np.ndarray], b: np.ndarray, c: np.ndarray, verbose: bool = False
):
    """
    Simplex Method Calculator

    Args:
        A (np.array[np.array]): Constraint Coefficient Matrix,  (n, m) shape
        b (np.array): Constraint Values,                        (n,) shape
        c (np.array): Objective Function Coefficients           (m,) shape
        verbose (bool): Whether to print intermediary states of A, pValues, and BasisCoeff
    """
    # Check types are correct
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    # Ensure Dimensions are Compatible
    assert A.shape[0] == b.shape[0]
    assert A.shape[1] == c.shape[0]
    artificialBasisIndex = np.where(c.imag != 0)
    basisIndex = np.where(
        (np.sum(A == 1, axis=0) == 1) & (np.sum(A == 0, axis=0) == A.shape[0] - 1)
    )[0]

    # if first column is unitary remove from basis,
    if (
        np.where(
            (np.sum(A.T[0] == 1, axis=0) == 1)
            & (np.sum(A.T[0] == 0, axis=0) == A.shape[0] - 1)
        )[0]
        == 0
    ):
        basisIndex = basisIndex[1:]
        # Get the index of the first 1 in each row
    first_indices = [
        np.where(row == 1)[0][0] if np.any(row == 1) else len(row)
        for row in A.T[basisIndex]
    ]

    # Get the order of rows based on the first occurrence of 1
    order = np.argsort(first_indices)
    basisIndex = basisIndex[order]
    pValueIndex = {}
    # bfs
    pValues = b
    while True:
        cBasisCoeff = c[basisIndex]
        maxP = np.dot(pValues, cBasisCoeff)
        evalVars = np.array(
            [np.dot(cBasisCoeff, A.T[i]) - c[i] for i in range(c.shape[0])]
        )
        evalVarsMagn = np.sum((evalVars.real, evalVars.imag * 1e16), axis=0)

        # If all Evaluation Variables are <= 0
        if (evalVarsMagn >= 0).all():
            # If Artificial Variable is leaving the basis and all values are greater than zero
            if np.isin(basisIndex, artificialBasisIndex).any():
                artBasis = basisIndex[
                    np.where(np.isin(basisIndex, artificialBasisIndex))
                ]
                feasibilityCheck = (A[:, artBasis] >= 0).all()
                if feasibilityCheck:
                    raise ValueError(np.where(artBasis == artificialBasisIndex)[0])
            pValues = {p + 1: pValues[i] for i, p in enumerate(basisIndex)}
            break

        if verbose:
            print("\nA:\n\n", A)
            print("\npValues:\n\n", pValues)
            print("\ncBasisCoeff:\n\n", cBasisCoeff)

        pivColIndex = np.argmin(evalVarsMagn)  # Entering
        negCheck = np.where(np.all(A[:, pivColIndex] <= 0, axis=0))[0]

        if negCheck.size != 0:
            raise OverflowError(pivColIndex)

        ratio = pValues / A.T[pivColIndex]
        ratio[ratio < 0] = np.inf
        pivRowIndex = np.argmin(ratio)  # Leaving
        pivCol = A.T[pivColIndex].copy()
        pivRow = A[pivRowIndex].copy()
        pivElement = A[pivRowIndex, pivColIndex]

        A[pivRowIndex] /= pivElement
        oldP = pValues.copy()
        p1 = pValues[pivRowIndex]

        pValues[pivRowIndex] /= pivElement
        pValueIndex[pivColIndex + 1] = pValues[pivRowIndex]
        if pivColIndex in basisIndex:
            c += -c[pivColIndex] * pivRow
        for i in range(A.shape[0]):
            if i != pivRowIndex:
                pValues[i] = (oldP[i] * pivElement - pivCol[i] * p1) / pivElement

                pValueIndex[i + 1] = pValues[i]
                for j in range(A.shape[1]):
                    A[i, j] = (
                        A[i, j] * pivElement - pivCol[i] * pivRow[j]
                    ) / pivElement

        basisIndex[pivRowIndex] = pivColIndex

    return maxP.real, pValues


def get_input():
    # Step 1: Get the number of variables
    while True:
        try:
            num_variables = int(input("Enter the number of variables: "))
            if num_variables <= 0:
                raise ValueError("The number of variables must be a positive integer.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid number.")

    # Step 2: Get the number of constraints
    while True:
        try:
            num_constraints = int(input("Enter the number of constraints: "))
            if num_constraints <= 0:
                raise ValueError(
                    "The number of constraints must be a positive integer."
                )
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid number.")

    constraints = []
    bounds = []

    # Objective Function
    print("Objective Function")
    while True:
        try:
            obj_coeff = list(
                map(
                    float,
                    input(
                        f"Enter the coefficients (space-separated) for variables (length {num_variables}): "
                    ).split(),
                )
            )
            if len(obj_coeff) != num_variables:
                raise ValueError
            break
        except ValueError:
            print("Invalid input: Please enter valid numeric coefficients.")

    # Step 3: Iterate through each constraint
    for i in range(num_constraints):
        print(f"\nConstraint {i + 1}:")

        # Get coefficients
        while True:
            try:
                coeff = list(
                    map(
                        float,
                        input(
                            f"Enter the coefficients (space-separated) for variables (length {num_variables}): "
                        ).split(),
                    )
                )
                if len(coeff) != num_variables:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input: Please enter valid numeric coefficients.")

        constraints.append(coeff)

        # Get the type of constraint
        while True:
            type = input("Enter the type of constraint (<=, >=, =): ").strip()
            if type in ["<=", ">=", "="]:
                break
            else:
                print("Invalid constraint type. Please enter <=, >=, or =.")

        # Get the right-hand side value
        while True:
            try:
                bound = float(input("Enter the right-hand side value: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid numeric value.")

        bounds.append((type, bound))

    return obj_coeff, constraints, bounds


def init_params(
    A, b, c, type
) -> tuple[np.ndarray[np.ndarray[float]], list[float], np.ndarray[float]]:
    A = np.array(A)
    n, _ = A.shape
    unit = np.eye(n)
    count = 0
    for i, _ in enumerate(unit):
        inq = b[i][0]
        unit = unit[:, ~np.all(unit == 0, axis=0)]
        unit[i] = np.zeros(unit.shape[1])

        if inq == ">=":
            temp_n = unit.shape[1]
            unit[i] = np.zeros(temp_n)

            slack = np.zeros(n)
            slack[i] = -1
            artificial = np.zeros(temp_n)
            artificial[i] = 1
            slack = np.array([slack, artificial])

            count += 1

        elif inq == "=":
            unit[i] = np.zeros(unit.shape[1])
            slack = np.zeros(n)
            artificial = np.zeros(temp_n)
            artificial[i] = 1
            slack = np.array([slack, artificial])
            count += 1

        elif inq == "<=":
            unit[i] = np.zeros(unit.shape[1])

            slack = np.zeros(n)
            slack[i] = 1
            slack = np.array([slack])

        unit = np.concatenate((unit, slack.T), axis=1)
    unit = unit[:, ~np.all(unit == 0, axis=0)]
    A = np.concatenate((A, unit), axis=1)
    b = [item[1] for item in b]
    c = np.concatenate((np.array(c), [0] * (A.shape[1] - len(c))), dtype=np.complex_)
    for i in range(count):
        c[-i - 1] = 0 + 1j

    if type == "min":
        c = -c
    return A, b, c


def start() -> None:
    # ---------------------------------------------------------------------------------------------------------------- #

    while True:
        type = str(
            input("Would you like to minimise or maximise? (max / min).")
        ).lower()
        if type in ["max", "min"]:
            break
    obj_coeff, constraints, bounds = get_input()
    A, b, c = init_params(constraints, bounds, obj_coeff, type)

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 1 - (Required Problem)

    # type = 'max'
    # A = np.array([[1., -1.,  2.,  0.,  1.,  1.,  0.,  0.,  0.,  1.],
    #               [0.,  1., -1.,  1.,  0.,  3.,  1.,  0.,  0.,  0.],
    #               [1.,  1., -3.,  1.,  1.,  0.,  0.,  1.,  0.,  0.],
    #               [1., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.]])
    # b = np.array([18., 8., 36., 23.])
    # c = np.array([2.+0.j, 3.+0.j, 4.+0.j, 1.+0.j, 8.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 2 - (All using the same constraints, taking from PS2 Answers since he switched the min/max)

    # A = np.array([[1., 2., -2., 4., 1., 0., 0.],
    #               [2., -1., 1., 2., 0., 1., 0.],
    #               [4., -2., 1., -1., 0., 0., 1.]])
    # b = np.array([40., 8., 10.])

    # 2a.
    # type = 'min'
    # c = np.array([2.+0.j, 1.+0.j, -3.+0.j, 5.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2b.
    # type = 'min'
    # c = np.array([3.+0.j, -1.+0.j, 3.+0.j, 4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2c.
    # type = 'max'
    # c = np.array([5.+0.j, -4.+0.j, 6.+0.j, -8.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2d.
    # type = 'max'
    # c = np.array([-4.+0.j, 6.+0.j, -2.+0.j, 4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 3 - (Unbounded)

    # type = 'max'
    # A = np.array([[1., -2.,  1.,  0.,  0.,  0.],
    #               [1.,  0.,  0.,  1.,  0.,  0.],
    #               [0.,  1.,  0.,  0., -1.,  1.]])

    # b = np.array([6., 10., 1.])

    # c = np.array([3.+0.j, 5.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 4 - (Unfeasible)

    # type = 'max'
    # A = np.array([[1., 1., 1., 0., 0.],
    #               [0., 1., 0., -1., 1,]])
    # b = np.array([5., 8.])
    # c = np.array([6.+0j, 4.+0j, 0.+0j, 0.+0j, 0.-1j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 5 - (Optimal Solution)

    type = "min"
    A = np.array(
        [
            [1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -1.0, 1.0],
        ]
    )
    b = np.array([6.0, 10.0, 1.0])
    c = np.array(
        [3.0 + 0.0j, 5.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j]
    )

    # Random example

    # type = 'min'
    # A = np.array([[-1.,  21., 903.,   1.,   1.,   0.,   0.,   0.],
    #               [100.,  32.,  10., -21.,   0.,  -1.,   1.,   0.],
    #               [3., 201., -23.,   0.,   0.,   0.,   0.,   1.]])

    # b = np.array([1.0, 100.0, 5.0])
    # c = np.array([-100.-0.j,   32.-0.j,  -90.-0.j,   -1.-0.j,   -0.-0.j,   -0.-0.j, -0.-1.j,   -0.-1.j])
    # ---------------------------------------------------------------------------------------------------------------- #

    # Leave this part alone :)

    try:
        result, coeff = simplex(A, b, c)
    except OverflowError as e:
        print(f"""\n\nError: There is a non-basic variable namely X{
            e.args[0]+1}, with all constraint coefficients non-positive. Thus the problem is unbounded.""")

    except ValueError as e:
        print(f"""\n\nError: There is a artificial variable namely A{
            e.args[0][0]+1} in the base with values greater than zero. Thus the problem is infeasible""")

    n, m = A.shape[0], A.shape[1] - A.shape[0]
    print("\n\nF* = ", result if type == "max" else -result)
    finalCoeff = {}
    for i in range(1, c.shape[0] + 1):
        if i not in coeff.keys():
            finalCoeff[i] = 0
        else:
            finalCoeff[i] = coeff[i]

    # Prints out the coefficients with the associated Variable names as they should appear
    coeefStr = [
        f"{f'X{i}'if i <= m else f'A{c.shape[0]-i+1}' if c[i-1].imag != 0 else f'S{
            i-m}'} = {finalCoeff[i]};"
        for i in range(1, c.shape[0] + 1)
    ]
    print(f"\nX* = ({coeefStr})")

    # ---------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    """ If using command line input, all Example variables should be commented out.

    If using Example premade inputs uncomment one set of them and comment out the block below.

    Comment Shortcut:   Ctrl + k + c
    Uncomment Shortcut: Ctrl + k + u
    """
    start()
