import numpy as np

# Function to generate a random tensor
def generate_random_tensor(l, m, n):
    return np.random.rand(l, m, n)

# Function to generate a tensor with user-provided values
def generate_custom_tensor(l, m, n):
    tensor = np.empty((l, m, n))
    print(f"Enter {l * m * n} elements for the {l} x {m} x {n} tensor:")
    for i in range(l):
        for j in range(m):
            for k in range(n):
                tensor[i, j, k] = float(input(f"Element ({i+1},{j+1},{k+1}): "))
    return tensor

# Main function to select the option and generate the tensor
def main():
    l = int(input("Enter the dimension l for the l x m x n tensor: "))
    m = int(input("Enter the dimension m for the l x m x n tensor: "))
    n = int(input("Enter the dimension n for the l x m x n tensor: "))

    choice = input("Do you want to generate a random tensor or input your own values? (Enter 'random' or 'custom'): ").strip().lower()
    
    if choice == 'random':
        tensor = generate_random_tensor(l, m, n)
    elif choice == 'custom':
        tensor = generate_custom_tensor(l, m, n)
    else:
        print("Invalid choice. Please enter 'random' or 'custom'.")
        return

    # ---------- Mode-1 Unfolding (j as leading axis) ----------
    tensor_unfolded_mode1 = []
    for i in range(l):
        row = []
        for j in range(m):
            for k in range(n):
                row.append(tensor[i][j][k])
        tensor_unfolded_mode1.append(row)

    # Perform SVD on mode-1 unfolded tensor
    matrix = tensor_unfolded_mode1

    # Compute the transpose of the matrix
    transpose_matrix = np.transpose(matrix)

    # Compute the product A^T A
    transpose_A_A = np.dot(transpose_matrix, matrix)

    # Determine the eigenvalues and eigenvectors of A^T A
    eigenvalues, eigenvectors = np.linalg.eigh(transpose_A_A)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(-np.abs(eigenvalues))
    sorted_eigenvalues = np.abs(eigenvalues[sorted_indices])
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Form the matrix V using the sorted eigenvectors
    V = sorted_eigenvectors

    # Compute the transpose of V (V^T)
    V_transpose = np.transpose(V)

    # Compute the singular values by taking the square root of the sorted eigenvalues
    singular_values = np.sqrt(sorted_eigenvalues)

    # Construct the diagonal matrix Σ by placing singular values along its diagonal
    Σ = np.diag(singular_values)

    # Compute the inverse of the diagonal matrix Σ
    epsilon = 1e-10
    Σ_inv = np.diag(1 / (singular_values + epsilon))

    # Compute U = AVΣ^(-1)
    U = np.dot(matrix, np.dot(V, Σ_inv))

    # Assign values for mode-1
    U1, S1, VT1 = U, singular_values, V_transpose
    
    # ---------- Mode-2 Unfolding (k as leading axis) ----------
    tensor_unfolded_mode2 = []
    for j in range(m):
        row = []
        for k in range(n):
            for i in range(l):
                row.append(tensor[i][j][k])
        tensor_unfolded_mode2.append(row)
    
    # Perform SVD on mode-2 unfolded tensor
    matrix = tensor_unfolded_mode2

    # Compute the transpose of the matrix
    transpose_matrix = np.transpose(matrix)

    # Compute the product A^T A
    transpose_A_A = np.dot(transpose_matrix, matrix)

    # Determine the eigenvalues and eigenvectors of A^T A
    eigenvalues, eigenvectors = np.linalg.eigh(transpose_A_A)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(-np.abs(eigenvalues))
    sorted_eigenvalues = np.abs(eigenvalues[sorted_indices])
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Form the matrix V using the sorted eigenvectors
    V = sorted_eigenvectors

    # Compute the transpose of V (V^T)
    V_transpose = np.transpose(V)

    # Compute the singular values by taking the square root of the sorted eigenvalues
    singular_values = np.sqrt(sorted_eigenvalues)

    # Construct the diagonal matrix Σ by placing singular values along its diagonal
    Σ = np.diag(singular_values)

    # Compute the inverse of the diagonal matrix Σ
    Σ_inv = np.diag(1 / (singular_values + epsilon))

    # Compute U = AVΣ^(-1)
    U = np.dot(matrix, np.dot(V, Σ_inv))

    # Assign values for mode-2
    U2, S2, VT2 = U, singular_values, V_transpose

    # ---------- Mode-3 Unfolding (i as leading axis) ----------
    tensor_unfolded_mode3 = []
    for k in range(n):
        row = []
        for i in range(l):
            for j in range(m):
                row.append(tensor[i][j][k])
        tensor_unfolded_mode3.append(row)

    # Perform SVD on mode-3 unfolded tensor
    matrix = tensor_unfolded_mode3

    # Compute the transpose of the matrix
    transpose_matrix = np.transpose(matrix)

    # Compute the product A^T A
    transpose_A_A = np.dot(transpose_matrix, matrix)

    # Determine the eigenvalues and eigenvectors of A^T A
    eigenvalues, eigenvectors = np.linalg.eigh(transpose_A_A)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(-np.abs(eigenvalues))
    sorted_eigenvalues = np.abs(eigenvalues[sorted_indices])
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Form the matrix V using the sorted eigenvectors
    V = sorted_eigenvectors

    # Compute the transpose of V (V^T)
    V_transpose = np.transpose(V)

    # Compute the singular values by taking the square root of the sorted eigenvalues
    singular_values = np.sqrt(sorted_eigenvalues)

    # Construct the diagonal matrix Σ by placing singular values along its diagonal
    Σ = np.diag(singular_values)

    # Compute the inverse of the diagonal matrix Σ
    Σ_inv = np.diag(1 / (singular_values + epsilon))

    # Compute U = AVΣ^(-1)
    U = np.dot(matrix, np.dot(V, Σ_inv))

    # Assign values for mode-3
    U3, S3, VT3 = U, singular_values, V_transpose

    # Remove the last n columns
    reduced_matrix_1 = U1[:, :n]
    reduced_matrix_2 = U2[:, :n]
    reduced_matrix_3 = U3[:, :n]

    # Get the factor matrices A, B, C
    A = reduced_matrix_1
    B = reduced_matrix_2
    C = reduced_matrix_3

    print("\nFactor Matrix A (from Mode-1 Unfolding):")
    print(A)

    print("\nFactor Matrix B (from Mode-2 Unfolding):")
    print(B)

    print("\nFactor Matrix C (from Mode-3 Unfolding):")
    print(C)

    # Redefining the tensor
    T = tensor
    
    # ---------- Mode-1 Unfolding ----------
    tensor_unfolded_mode1 = []
    for i in range(l):
        row = []
        for j in range(m):
            for k in range(n):
                row.append(T[i][j][k])
        tensor_unfolded_mode1.append(row)

    T1 = tensor_unfolded_mode1
    AT = np.transpose(A)

    m_mult, n_cols = len(AT), len(AT[0])
    n_rows, p = len(T1), len(T1[0])

    # Resultant matrix of size m by p initialized with zeros
    result = [[0 for _ in range(p)] for _ in range(m_mult)] 

    # Matrix multiplication
    for i in range(len(AT)):
        for j in range(len(T1[0])):
            for k in range(len(T1)):
                result[i][j] += AT[i][k] * T1[k][j]

    T1result = result

    # ---------- Mode-1 Refolding ----------
    reconstructed1 = [[[0 for _ in range(n)] for _ in range(m)] for _ in range(len(AT))]
    for i in range(len(AT)):
        for j in range(m):
            for k in range(n):
                flat_index = j * n + k
                reconstructed1[i][j][k] = T1result[i][flat_index]

    T11 = reconstructed1
    
    # ---------- Mode-2 Unfolding (corrected for k as leading axis) ----------
    tensor_unfolded_mode2 = []
    for j in range(m):
        row = []
        for k in range(n):
            for i in range(len(AT)):
                row.append(T11[i][j][k])
        tensor_unfolded_mode2.append(row)

    T2 = tensor_unfolded_mode2
    BT = np.transpose(B)

    m_mult, n_cols = len(BT), len(BT[0])
    n_rows, p = len(T2), len(T2[0])

    result2 = [[0 for _ in range(p)] for _ in range(m_mult)] 

    # Matrix multiplication
    for i in range(len(BT)):
        for j in range(len(T2[0])):
            for k in range(len(T2)):
                result2[i][j] += BT[i][k] * T2[k][j]

    T2result = result2

    # ---------- Mode-2 Refolding (corrected) ----------
    reconstructed2 = [[[0 for _ in range(n)] for _ in range(len(BT))] for _ in range(len(AT))]
    for j in range(len(BT)):
        for k in range(n):
            for i in range(len(AT)):
                flat_index = k * len(AT) + i
                reconstructed2[i][j][k] = T2result[j][flat_index]

    T22 = reconstructed2
    
    # ---------- Mode-3 Unfolding (corrected for i as leading axis) ----------
    tensor_unfolded_mode3 = []
    for k in range(n):
        row = []
        for i in range(len(AT)):
            for j in range(len(BT)):
                row.append(T22[i][j][k])
        tensor_unfolded_mode3.append(row)

    T3 = tensor_unfolded_mode3
    CT = np.transpose(C)

    m_mult, n_cols = len(CT), len(CT[0])
    n_rows, p = len(T3), len(T3[0])

    result3 = [[0 for _ in range(p)] for _ in range(m_mult)] 
    
    # Matrix multiplication
    for i in range(len(CT)):
        for j in range(len(T3[0])):
            for k in range(len(T3)):
                result3[i][j] += CT[i][k] * T3[k][j]

    T3result = result3
    
    # ---------- Mode-3 Refolding (corrected) ----------
    reconstructed3 = []
    for i in range(len(AT)):
        row = []
        for j in range(len(BT)):
            col = []
            for k in range(len(CT)):
                flat_index = i * len(BT) + j
                col.append(T3result[k][flat_index])
            row.append(col)
        reconstructed3.append(row)

    G = reconstructed3
    
    print("\nCore Tensor G:")
    for i in range(len(AT)):
        print(f"Slice {i}:")
        for row in reconstructed3[i]:
            print([float(x) for x in row])

    # Reconstruction verification (calculations only, no prints)
    tensor_unfolded_mode1 = []
    for i in range(len(AT)):
        row = []
        for j in range(len(BT)):
            for k in range(len(CT)):
                row.append(G[i][j][k])
        tensor_unfolded_mode1.append(row)

    G1 = tensor_unfolded_mode1

    m_mult, n_cols = len(A), len(A[0])
    n_rows, p = len(G1), len(G1[0])

    result_recon1 = [[0 for _ in range(p)] for _ in range(m_mult)] 

    # Matrix multiplication
    for i in range(len(A)):
        for j in range(len(G1[0])):
            for k in range(len(G1)):
                result_recon1[i][j] += A[i][k] * G1[k][j]

    G1result = result_recon1

    # ---------- Mode-1 Refolding ----------
    reconstructed1 = [[[0 for _ in range(len(CT))] for _ in range(len(BT))] for _ in range(l)]
    for i in range(l):
        for j in range(len(BT)):
            for k in range(len(CT)):
                flat_index = j * len(CT) + k
                reconstructed1[i][j][k] = G1result[i][flat_index]

    G11 = reconstructed1
    
    # ---------- Mode-2 Unfolding (corrected for k as leading axis) ----------
    tensor_unfolded_mode2 = []
    for j in range(len(BT)):
        row = []
        for k in range(len(CT)):
            for i in range(l):
                row.append(G11[i][j][k])
        tensor_unfolded_mode2.append(row)

    G2 = tensor_unfolded_mode2

    m_mult, n_cols = len(B), len(B[0])
    n_rows, p = len(G2), len(G2[0])

    result_recon2 = [[0 for _ in range(p)] for _ in range(m_mult)] 
    
    # Matrix multiplication
    for i in range(len(B)):
        for j in range(len(G2[0])):
            for k in range(len(G2)):
                result_recon2[i][j] += B[i][k] * G2[k][j]

    G2result = result_recon2

    # ---------- Mode-2 Refolding (corrected) ----------
    reconstructed2 = [[[0 for _ in range(len(CT))] for _ in range(m)] for _ in range(l)]
    for j in range(m):
        for k in range(len(CT)):
            for i in range(l):
                flat_index = k * l + i
                reconstructed2[i][j][k] = G2result[j][flat_index]

    G22 = reconstructed2
    
    # ---------- Mode-3 Unfolding (corrected for i as leading axis) ----------
    tensor_unfolded_mode3 = []
    for k in range(len(CT)):
        row = []
        for i in range(l):
            for j in range(m):
                row.append(G22[i][j][k])
        tensor_unfolded_mode3.append(row)

    G3 = tensor_unfolded_mode3

    m_mult, n_cols = len(C), len(C[0])
    n_rows, p = len(G3), len(G3[0])
    
    result_recon3 = [[0 for _ in range(p)] for _ in range(m_mult)] 
    
    # Matrix multiplication
    for i in range(len(C)):
        for j in range(len(G3[0])):
            for k in range(len(G3)):
                result_recon3[i][j] += C[i][k] * G3[k][j]

    G3result = result_recon3
    
    # ---------- Mode-3 Refolding (corrected) ----------
    reconstructed3 = [[[0 for _ in range(n)] for _ in range(m)] for _ in range(l)]
    for k in range(n):
        for i in range(l):
            for j in range(m):
                flat_index = i * m + j
                reconstructed3[i][j][k] = G3result[k][flat_index]

    T_reconstructed = reconstructed3

    # Verification (calculations only)
    max_diff = 0.0
    for i in range(l):
        for j in range(m):
            for k in range(n):
                diff = abs(tensor[i][j][k] - T_reconstructed[i][j][k])
                max_diff = max(max_diff, diff)
    
    if max_diff < 1e-5:
        print("\n✓ SUCCESS: Since the reconstructed tensor is equal to the original tensor, Tucker decomposition is verified!")
    else:
        print(f"\n⚠ WARNING: Maximum difference: {max_diff}")
       
if __name__ == "__main__":
    main()
