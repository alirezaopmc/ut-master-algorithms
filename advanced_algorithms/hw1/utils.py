from functools import wraps
import random
import time
import numpy as np
from typing import Callable, List, Tuple


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if this is the root call by looking for an attribute
        if not hasattr(wrapper, "is_root") or getattr(wrapper, "is_root") is False:
            wrapper.is_root = True  # Mark the start of the root call
            start_time = time.time_ns()
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time_ns()
            print(f"Elapsed time: {end_time - start_time:.4f} nanoseconds")
            wrapper.is_root = False  # Reset after the root call is complete
            return result
        else:
            return func(*args, **kwargs)  # Normal execution for recursive calls

    return wrapper


def measure_time(func, *args, **kwargs):
    start_time = time.time_ns()
    func(*args, **kwargs)
    end_time = time.time_ns()
    elapsed_time = end_time - start_time
    return elapsed_time


def generate_sorted_number_pairs(N: int, min_int: int, max_int: int) -> List[int]:
    if N > (max_int - min_int + 1):
        raise ValueError("N must be less than or equal to the range of numbers.")
    Xs = random.sample(range(min_int, max_int + 1), N)
    Xs.sort()
    Ys = random.sample(range(min_int, max_int + 1), N)
    Ys.sort()

    return [(Xs[i], Ys[i]) for i in range(N)]


def generate_sorted_square_matrix_pairs(
    N: int, min_int: int, max_int: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Generate random sizes for square matrices, ensuring they are powers of two
    powers_of_two = [
        2**i for i in range(int(np.log2(min_int)), int(np.log2(max_int)) + 1)
    ]
    matrix_sizes = random.sample(powers_of_two, N)

    # Generate random square matrices of those sizes
    matrices = []
    for size in matrix_sizes:
        A = np.random.randint(min_int, max_int + 1, (size, size))
        B = np.random.randint(min_int, max_int + 1, (size, size))
        matrices.append((A, B))

    # Sort matrices by their total number of elements (size * size)
    matrices.sort(key=lambda matrix: matrix[0].shape[0])

    return matrices


def sample_int_mult_algorithm(
    mult_func: Callable[[int, int], int], pairs: List[Tuple[int, int]]
) -> List[int]:
    results = []
    for x, y in pairs:
        result = measure_time(mult_func, x, y)
        results.append(result)
    return results


def sample_matrix_mult_algorithm(
    matrix_mult_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> List[int]:
    results = []
    for x, y in pairs:
        result = measure_time(matrix_mult_func, x, y)
        results.append(result)
    return results
