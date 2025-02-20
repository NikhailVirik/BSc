import numpy as np 

def generate_jump_op(row: int, column: int) -> np.ndarray:
    "Generate jump operator by defining position in the matrix"

    init = np.zeros((4,4))
    init[row,column] = 1
    return init

jump_op_10 = generate_jump_op(0, 3)
jump_op_p0 = generate_jump_op(1, 3)
jump_op_m0 = generate_jump_op(2, 3)
jump_op_1m = generate_jump_op(0, 2)
jump_op_pm = generate_jump_op(1, 2)
jump_op_1p = generate_jump_op(0, 1)

a = jump_op_10 @ np.zeros((4,4))