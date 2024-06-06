import sympy as sp


mat = sp.Matrix([[0, 11, 0, 0, 0],
      [11, 0, -3, 27, -3],
      [0, -3, 0, 6, 28],
      [0, 27, 6, 0, 0],
      [0, -3, 28, 0, 0]])
print(mat)
print(mat.inv())

# Flatten the list of lists into a single list (if intended)
flat_lu = [item for sublist in lu for item in sublist]
print(flat_lu)
