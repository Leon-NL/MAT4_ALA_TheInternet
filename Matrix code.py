import numpy as np

# Set the dimensions of the matrix
rows, cols = 4, 4
    
# Create a matrix with random 1's and 0's
matrix = np.random.randint(2, size=(rows, cols))

# Set the diagonal elements to zero
np.fill_diagonal(matrix, 0)

# Calculate the sum of the 1's in each column
matrix_sum = matrix.sum(axis=0)

# Avoid division by zero by replacing zero sums with ones temporarily
matrix_sum_safe = np.where(matrix_sum == 0, 1, matrix_sum)

# Divide each element by the sum of the 1's in its column
normalized_matrix = matrix / matrix_sum_safe

# Print results
print("Original Matrix:")
print(matrix)
print("\nSum of 1's in Each Column:")
print(matrix_sum)
print("\nNormalized Matrix:")
print(normalized_matrix)  
# type: ignore