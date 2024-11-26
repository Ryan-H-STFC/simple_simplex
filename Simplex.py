from typing import Literal
import numpy as np
import readchar
from rich import print


def simplex(
    A: np.ndarray, b: np.ndarray, c: np.ndarray, verbose: bool = False
) -> tuple[float, dict[float]]:
    """
    ``simplex``
    -----------
    Simplex Method Calculator, making use of complex numbers to represent the Big M in calculations.

    Args:
        A (np.array): Constraint Coefficient Matrix,            (n, m) shape
        b (np.array): Constraint Values,                        (n,) shape
        c (np.array): Objective Function Coefficients,          (m,) shape
        verbose (bool): Whether to print intermediary states of A, pValues, and BasisCoeff. Defaults to False.

    Raises:
        OverflowError: Used when the problem is unbounded.
        ValueError: Used when the problem is infeasible.

    Returns:
        tuple[float, dict[str, float]]: Returns both the max/min value, along with a dictionary containing the values
        of each variable.
    """

    # Check types are correct
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=np.float64)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=np.float64)
    if not isinstance(c, np.ndarray):
        c = np.array(c, dtype=np.complex128)

    # Ensure Dimensions are Compatible
    assert A.shape[0] == b.shape[0]
    assert A.shape[1] == c.shape[0]

    # Artificial Variables use imaginary values to represent Big M
    artificialBasisIndex = np.where(c.imag != 0)

    # Basis Index finds columns that are standard, i.e. artificial/slack with single 1 and zeros elsewhere
    basisIndex = np.where(
        (np.sum(A == 1, axis=0) == 1) & (np.sum(A == 0, axis=0) == A.shape[0] - 1)
    )[0]
    basicIndex = np.array(range(A.shape[1] - A.shape[0]))

    # Remove basic variables from basis if present, case where basic variables are standard
    basisIndex = basisIndex[~basicIndex]

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
    pValues = b.copy()
    while True:

        cBasisCoeff = c[basisIndex]
        maxP = np.dot(pValues, cBasisCoeff)
        evalVars = np.array(
            [np.dot(cBasisCoeff, A.T[i]) - c[i] for i in range(c.shape[0])]
        )
        # Magnitude of evaluation variables,
        evalVarsMagn = np.sum((evalVars.real, evalVars.imag * 1e16), axis=0)

        if verbose:
            print("\nA:\n\n", A)
            print("\npValues:\n\n", pValues)
            print("\ncBasisCoeff:\n\n", cBasisCoeff)

        # Entering Variable
        pivColIndex = np.argmin(evalVarsMagn)

        # Unboundedness check, if all values in the pivot column are less than or equal to zero the problem is unbounded
        if (A[:, pivColIndex] <= 0).all():
            finalCoeff = np.zeros(c.shape[0])
            finalCoeff[basisIndex] = pValues
            m = c[c.real > 0].shape[0]

            coeffStr = [
                f"{f'X{i+1}' if i < m else f'A{c.shape[0]-i}' if c[i].imag != 0 else f'S{i-m + 1}'}"
                for i in range(0, c.shape[0])
            ]
            raise OverflowError(coeffStr[pivColIndex])

        # Infeasibility check
        # If all Evaluation Variables are greater then or equal to 0
        if (evalVarsMagn >= 0).all():
            # If Artificial Variable are contained in the final solution the problem is infeasible

            artBasis = basisIndex[
                np.where(np.isin(basisIndex, artificialBasisIndex))
            ]
            infeasibilityCheck = (A[:, artBasis] > 0).any()
            if infeasibilityCheck:
                raise ValueError(np.where(artBasis == artificialBasisIndex)[0])
            pValues = {p + 1: pValues[i] for i, p in enumerate(basisIndex)}
            break

        # Find Ratio of RHS values with pivot column
        # Ignore divide by zero error, results become infinite
        with np.errstate(divide='ignore'):
            ratio = np.divide(pValues, A.T[pivColIndex])

        # Ignore negatives by setting at infinity
        ratio[ratio < 0] = np.inf

        # Leaving Variable is minimum positive of the ratios
        pivRowIndex = np.argmin(ratio[np.where(ratio > 0)])
        pivCol = A.T[pivColIndex].copy()
        pivRow = A[pivRowIndex].copy()
        pivElement = A[pivRowIndex, pivColIndex]

        # Normalise pivot row
        A[pivRowIndex] /= pivElement
        oldP = pValues.copy()
        p1 = pValues[pivRowIndex]

        # Calculate Next Tableau Iteration
        pValues[pivRowIndex] = pValues[pivRowIndex] / pivElement
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
        # Entering replaces Leaving variable
        basisIndex[pivRowIndex] = pivColIndex

    return maxP.real, pValues


def get_input() -> tuple[list[float], list[list[float]], tuple[str, float]]:
    """
    ``get_input``
    -------------

    Raises:
        ValueError: Invalid input, requires positive int for number of variables.
        ValueError: Invalid input, requires positive int for number of constaints.
        ValueError: Invalid input, requires numerical value for constraint coefficients.
        ValueError: Invalid input, requires numerical value for RHS.

    Returns:
        tuple[list[float], list[list[float]], tuple[str, float]]:
        Returns the inputs Objective function coefficients,
        the Constraint matrix,
        and the bounds along with constraint type.

    """

    # Step 1: Get the number of variables
    while True:
        try:
            num_variables = int(input("\nEnter the number of variables: "))
            if num_variables <= 0:
                raise ValueError("\nThe number of variables must be a positive integer.\n")
            break
        except ValueError as e:
            print(f"\nInvalid input: {e}Please enter a valid number.\n")

    # Step 2: Get the number of constraints
    while True:
        try:
            num_constraints = int(input("\nEnter the number of constraints: "))
            if num_constraints <= 0:
                raise ValueError(
                    "\nThe number of constraints must be a positive integer.\n"
                )
            break
        except ValueError as e:
            print(f"\nInvalid input: {e}Please enter a valid number.\n")

    constraints = []
    bounds = []

    # Objective Function
    print("\nObjective Function")
    while True:
        try:
            obj_coeff = list(
                map(
                    float,
                    input(
                        f"\nEnter the objective coefficients (space-separated) for variables (length {num_variables}): "
                    ).split(),
                )
            )
            if len(obj_coeff) != num_variables:
                raise ValueError
            break
        except ValueError:
            print("\nInvalid input: Please enter valid numeric coefficients.\n")

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
                            f"\nEnter the coefficients (space-separated) for variables (length {num_variables}): "
                        ).split(),
                    )
                )
                if len(coeff) != num_variables:
                    raise ValueError
                break
            except ValueError:
                print("\nInvalid input: Please enter valid numeric coefficients.\n")

        constraints.append(coeff)

        # Get the type of constraint
        while True:
            inq = input("\nEnter the type of constraint (<=, >=, =): ").strip()
            if inq in ["<=", ">=", "="]:
                break
            else:
                print("\nInvalid constraint type. Please enter <=, >=, or =.\n")

        # Get the right-hand side value
        while True:
            try:
                bound = float(input("\nEnter the right-hand side value: "))
                break
            except ValueError:
                print("\nInvalid input. Please enter a valid numeric value.\n")

        bounds.append((inq, bound))

    return obj_coeff, constraints, bounds


def init_params(
    A: np.ndarray, b: np.ndarray, c: np.ndarray, which: Literal['max', 'min']
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """
    ``init_params``
    ---------------

    Args:
        A (np.ndarray): Constraint Coefficient Matrix
        b (np.ndarray): Constraint Values
        c (np.ndarray): Objective Function Coefficients
        which (Literal[&#39;max&#39;, &#39;min&#39;]): string of 'max' or 'min'. Whether to maximise or minimise.

    Returns:
        tuple[np.ndarray, list[float], np.ndarray]: A now has slack and artificial variables and has handled negative
        constraints, b returned as just the values of the constraints after handling negatives, c is prepared with slack
        and artificial variables.
    """
    A = np.array(A)
    n, _ = A.shape
    unit = np.zeros(n)
    count = 0

    # Handle negative constraint values
    for i, (inq, value) in enumerate(b):
        if value < 0:
            A[i] *= -1
            new_value = value * -1
            if inq == '<=':
                new_inq = '>='
            elif inq == '>=':
                new_inq = '<='
            else:
                new_inq = '='
            b[i] = (new_inq, new_value)

    dim = A.shape[0]
    artificial = np.zeros([dim, 1])
    slack = np.zeros([dim, 1])

    # Prep objective coefficients for artificial and slack variables
    c = np.concatenate((np.array(c), [0]*A.shape[0]), dtype=np.complex_)

    # Loop will add slack if <=,
    # subtract slack and add artificial if >=,
    # add artificial if =
    for i in range(A.shape[0]):

        inq = b[i][0]
        newArt = np.zeros([dim, 1])
        newSlack = np.zeros([dim, 1])
        if inq == '<=':
            newSlack[i] = 1
            slack = np.concatenate([slack, newSlack], axis=1)

        if inq == '=':
            newArt[i] = 1
            artificial = np.concatenate([artificial, newArt], axis=1)
            count += 1

        if inq == '>=':
            newSlack[i] = -1
            newArt[i] = 1
            slack = np.concatenate([slack, newSlack], axis=1)
            artificial = np.concatenate([artificial, newArt], axis=1)
            c = np.concatenate((np.array(c), [0]), dtype=np.complex_)
            count += 1
    unit = np.concatenate([slack, artificial], axis=1)

    # Remove zero columns
    unit = unit[:, ~np.all(unit == 0, axis=0)]
    # Add slack and artificial variables to Constraints matrix
    A = np.concatenate((A, unit), axis=1)
    b = [item[1] for item in b]

    # Add -M represented as imaginary number for each time an artificial variable was added.
    for i in range(count):
        c[-i-1] = 0-1j

    # If minimising swap parity of objective function coefficients
    if which == "min":
        c = -c
    return A, b, c


def start() -> None:
    """
    If using command line input, all Example variables should be commented out.

    If using Example premade inputs uncomment one set of them and comment out the block below which takes inputs.

    VSCode Keyboard Shortcuts:
    Comment Shortcut:   Ctrl + k + c
    Uncomment Shortcut: Ctrl + k + u
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # This block takes input for any problem the user wants

    # while True:
    #     which = str(
    #         input("\nWould you like to minimise or maximise? (max / min): ")
    #     ).lower()
    #     if which in ["max", "min"]:
    #         break
    # obj_coeff, constraints, bounds = get_input()

    # A, b, c = init_params(constraints, bounds, obj_coeff, which)

    # ---------------------------------------------------------------------------------------------------------------- #

    # Required Problem
    # F* = 232.0
    # X* = X₁ = 0; X₂ = 8.0; X₃ = 0; X₄ = 0; X₅ = 26.0; X₆ = 0; S₁ = 0; S₂ = 2.0; S₃ = 5.0; A₁ = 0;

    # ---------------------------------------------------------------------------------------------------------------- #

    which = 'max'
    A = np.array([[1., -1.,  2.,  0.,  1.,  1.,  0.,  0.,  0.,  1.],
                  [0.,  1., -1.,  1.,  0.,  3.,  1.,  0.,  0.,  0.],
                  [1.,  1., -3.,  1.,  1.,  0.,  0.,  1.,  0.,  0.],
                  [1., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.]])
    b = np.array([18., 8., 36., 23.])
    c = np.array([2.+0.j, 3.+0.j, 4.+0.j, 1.+0.j, 8.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 2 - (All using the same constraints, taking from PS2 Answers since all min)

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'min'
    # A = np.array([[1., 2., -2., 4., 1., 0., 0.],
    #               [2., -1., 1., 2., 0., 1., 0.],
    #               [4., -2., 1., -1., 0., 0., 1.]])
    # b = np.array([40., 8., 10.])

    # 2a.       F* = -41,   X* = X₁ = 0; X₂ = 6.0; X₃ = 0; X₄ = 7.0; S₁ = 0; S₂ = 0; S₃ = 29.0;

    # c = -np.array([-2.+0.j, -1.+0.j, 3.+0.j, -5.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2b.       (Unbounded) on X3

    # c = -np.array([-3.+0.j, +1.+0.j, -3.+0.j, -4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2c.       F* = -80,   X* = X₁ = 0; X₂ = 6.0; X₃ = 0; X₄ = 7.0; S₁ = 0; S₂ = 0; S₃ = 29.0;

    # c = -np.array([5.+0.j, -4.+0.j, 6.+0.j, -8.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # 2d.       F* = -16.0, X* = X₁ = 1.0; X₂ = 0; X₃ = 6.0; X₄ = 0; S₁ = 51.0; S₂ = 0; S₃ = 0;

    # c = -np.array([-4.+0.j, 6.+0.j, -2.+0.j, 4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 3 - (Unbounded)

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'max'
    # A = np.array([[1., -2.,  1.,  0.,  0.,  0.],
    #               [1.,  0.,  0.,  1.,  0.,  0.],
    #               [0.,  1.,  0.,  0., -1.,  1.]])

    # b = np.array([6., 10., 1.])

    # c = np.array([3.+0.j, 5.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 4 - (Unbounded)

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = "min"
    # A = np.array(
    #     [
    #         [1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, 0.0, -1.0, 1.0],
    #     ]
    # )
    # b = np.array([6.0, 10.0, 1.0])
    # c = np.array(
    #     [3.0 + 0.0j, 5.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j]
    # )

    # ---------------------------------------------------------------------------------------------------------------- #

    # Example Problem 5 - (Infeasible)

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'max'
    # A = np.array([[1., 1., 1., 0., 0.],
    #               [0., 1., 0., -1., 1,]])
    # b = np.array([5., 8.])
    # c = np.array([6.+0j, 4.+0j, 0.+0j, 0.+0j, 0.-1j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Random example

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'min'
    # A = np.array([[-1.,  21., 903.,   1.,   1.,   0.,   0.,   0.],
    #               [100.,  32.,  10., -21.,   0.,  -1.,   1.,   0.],
    #               [3., 201., -23.,   0.,   0.,   0.,   0.,   1.]])

    # b = np.array([1.0, 100.0, 5.0])
    # c = np.array([-100.-0.j,   32.-0.j,  -90.-0.j,   -1.-0.j,   -0.-0.j,   -0.-0.j, -0.-1.j,   -0.-1.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Big Example -

    # F* = 384.4444444444444
    # X* = X₁ = 0; X₂ = 0; X₃ = 0; X₄ = 12.222222222222221; X₅ = 4.444444444444445; X₆ = 0; X₇ = 28.14814814814815;
    # X₈ = 26.296296296296294; X₉ = 0; X₁0 = 0;
    # S₁ = 0; S₂ = 0; S₃ = 35.925925925925924; S₄ = 40.0; # S₅ = 11.851851851851855; S₆ = 50.370370370370374; S₇ = 0;
    # S₈ = 0;

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'max'
    # A = np.array([[1., 2., 1., 3., 2., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    #               [2., 1., 3., 1., 1., 2., 1., 4., 1., 3., 0., 1., 0., 0., 0., 0., 0., 0.],
    #               [3., 3., 2., 4., 1., 1., 3., 1., 1., 2., 0., 0., 1., 0., 0., 0., 0., 0.],
    #               [4., 2., 2., 1., 3., 2., 1., 1., 2., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
    #               [2., 1., 4., 3., 2., 4., 2., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
    #               [5., 1., 2., 2., 1., 3., 1., 2., 1., 3., 0., 0., 0., 0., 0., 1., 0., 0.],
    #               [3., 4., 1., 2., 5., 2., 1., 4., 1., 2., 0., 0., 0., 0., 0., 0., 1., 0.],
    #               [1., 1., 1., 1., 2., 3., 2., 2., 4., 3., 0., 0., 0., 0., 0., 0., 0., 1.]])
    # b = np.array([100., 150., 200., 120., 140., 160., 180., 130.])
    # c = np.array([3.+0.j, 4.+0.j, 2.+0.j, 5.+0.j, 6.+0.j, 2.+0.j, 4.+0.j, 7.+0.j, 3.+0.j,
    #              4.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

    # ---------------------------------------------------------------------------------------------------------------- #

    # Standard Example

    # ---------------------------------------------------------------------------------------------------------------- #

    # which = 'max'
    # A = np.array([[1., 0., 0., 1., 0., 0.],
    #               [0., 1., 0., 0., 1., 0.],
    #               [0., 0., 1., 0., 0., 1.]])
    # b = np.array([5.0, 23.0, 3.0])
    # c = np.array([2.+0.j, 3.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    # ---------------------------------------------------------------------------------------------------------------- #

    # Leave this part alone :)

    # function to convert to subscript (Do not mark)

    def get_sub(x: str) -> str:
        """
        Do not mark, I am using this simply for a nicer output.
        https://www.geeksforgeeks.org/how-to-print-superscript-and-subscript-in-python/

        Args:
            x (str): str to translate to subscript

        Returns:
            str: subscripted str of x
        """
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
        res = x.maketrans(''.join(normal), ''.join(sub_s))
        return x.translate(res)

    try:
        result, coeff = simplex(A, b, c)
    except OverflowError as e:
        print(f"\n\nError: There is a non-basic variable namely {e.args[0][0]}{get_sub(str(e.args[0][1:]))}",
              "with all constraint coefficients non-positive. Thus the problem is unbounded.\n\n")
        exit()

    except ValueError as e:
        print(f"\n\nError: There is a artificial variable namely A{get_sub(str(e.args[0][0]+1))} in the basis with",
              "values greater than zero. Thus the problem is infeasible.\n\n")
        exit()
    finalCoeff = np.zeros(c.shape[0])
    finalCoeff = [coeff.get(i, 0) for i in range(1, c.shape[0]+1)]

    m = c[c.real != 0].shape[0]
    print("\n\nF* =", result if which == "max" else -result)

    # Prints out the coefficients with the associated Variable names as they should appear
    coeefStr = [
        f"{f'X{i}'if i <= m else f'A{c.shape[0]-i+1}' if c[i-1].imag != 0 else f'S{
            i-m}'} = {finalCoeff[i-1]};"
        for i in range(1, c.shape[0] + 1)
    ]

    print("\nX* =", end=' ')
    for var in coeefStr:
        print(f"{var[0]}{get_sub(var[1])}{var[2:]}", end=' ')
    print("\n\n")

    # ---------------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    start()
    print("\n\n\n\nPress any key to exit...")
    readchar.readchar()
