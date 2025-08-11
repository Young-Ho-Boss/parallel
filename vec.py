from math import sqrt, ceil, log2

# 행렬 -> 벡터 (row-major)
def flatten_matrix(A: list) -> list:
    vector = []
    m = len(A)
    n = len(A[0])
    for i in range(m):
        for j in range(n):
            vector.append(A[i][j])
    return vector

# 벡터 -> 행렬 (row-major)
def unflatten_vector(v: list, m: int, n: int) -> list:
    return [v[i*n:(i+1)*n] for i in range(m)]

# 보기 좋게 출력
def print_matrix(M):
    for row in M:
        print(" ".join(f"{x:8}" for x in row))

def sigma(x):
    D = len(x[0])
    return [[x[i][(i + j) % D] for j in range(D)] for i in range(D)]

def tau(x):
    D = len(x[0])
    return [[x[(i + j) % D][j] for j in range(D)] for i in range(D)]

def psi(x,k):
    D = len(x[0])
    return [[x[(i+k)%D][(j)] for j in range(D)] for i in range(D)]

def pi(x,k):
    D = len(x[0])
    return [[x[(i)][(j+k)%D] for j in range(D)] for i in range(D)]

def hadamard(A, B):
    D = len(A)
    return [[A[i][j] * B[i][j] for j in range(D)] for i in range(D)]
    
def he_matmul_plain(A, B):
    # Step 1-1, 1-2
    SA = sigma(A)     # σ(A)
    TB = tau(B)       # τ(B)

    D = len(A)
    # Step 3 누적자
    C = [[0 for _ in range(D)] for _ in range(D)]
    count = 0
    
    # Step 2: k=0..D-1
    for k in range(D):
        Xk = pi(SA, k)   # φ^k(σ(A))  : 열 방향 회전
        Yk = psi(TB, k)  # ψ^k(τ(B))  : 행 방향 회전
        Pk = hadamard(Xk, Yk)  # 슬롯별 곱
        # 누적
        for i in range(D):
            for j in range(D):
                C[i][j] += Pk[i][j]
                count += 1
    return C, count


if __name__ == "__main__":
    D = 9
    A = [[D*i + j + 1 for j in range(D)] for i in range(D)] 
    I = [[1 if i == j else 0 for j in range(D)] for i in range(D)]
    v = [A[i][j] for i in range(D) for j in range(D)]
    u_tau_k = [1 if (ell % D) == D-1 else 0 for ell in range(D*D)]
    print("그냥 행렬 A")
    print_matrix(A)
    print("\n")
    print("sigma")
    print_matrix(sigma(A))
    print("\n")
    print("tau")
    print_matrix(tau(A))
    print("\n")
    print("pi")
    print_matrix(pi(sigma(A),3))
    print("\n")
    print("psi")
    print_matrix(psi(tau(A),3))
    # print(u_tau_k)
    C, count = he_matmul_plain(A, A)
    print("C = A*A (슬롯 레이아웃 기준)")
    print_matrix(C)
    print(f"count: {count}")