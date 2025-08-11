from typing import List
from math import ceil, isqrt

def print_matrix(M: List[List[int]]):
    for row in M:
        print(" ".join(f"{x:8}" for x in row))

class ScalarOpCounter:
    def __init__(self):
        self.scalar_adds = 0      # 스칼라 덧셈 (Hadamard 누적)
        self.scalar_mults = 0     # 스칼라 곱   (Hadamard)
        self.scalar_moves = 0     # 원소 이동(대입) 횟수 = 재배열에 의한 D^2 대입들
        self.calls_pi = 0         # 열방향 재배열 호출 수 (논리적 회전 수 추정용)
        self.calls_psi = 0        # 행방향 재배열 호출 수

    # 재배열을 실제로 "물리적"으로 만들어낼 때만 move 카운트 증가
    def move_matrix(self, src, fetch):
        D = len(src)
        dst = [[0 for _ in range(D)] for _ in range(D)]
        for i in range(D):
            for j in range(D):
                dst[i][j] = fetch(i, j)
                self.scalar_moves += 1
        return dst

    def hadamard(self, A, B):
        D = len(A)
        C = [[0 for _ in range(D)] for _ in range(D)]
        for i in range(D):
            for j in range(D):
                C[i][j] = A[i][j] * B[i][j]
                self.scalar_mults += 1
        return C

    def accumulate(self, C, P):
        D = len(C)
        for i in range(D):
            for j in range(D):
                C[i][j] += P[i][j]
                self.scalar_adds += 1

# ---- 변환 함수들 (재배열을 실제로 생성할 때만 move를 센다) ----
def sigma(x, ops: ScalarOpCounter):
    D = len(x[0])
    # [σ(x)]_{i,j} = x_{i, (i+j) mod D}
    return ops.move_matrix(x, lambda i, j: x[i][(i + j) % D])

def tau(x, ops: ScalarOpCounter):
    D = len(x[0])
    # [τ(x)]_{i,j} = x_{(i+j) mod D, j}
    return ops.move_matrix(x, lambda i, j: x[(i + j) % D][j])

def pi(x, k, ops: ScalarOpCounter):
    D = len(x[0])
    ops.calls_pi += 1
    # [π(x,k)]_{i,j} = x_{i, (j+k) mod D}
    return ops.move_matrix(x, lambda i, j: x[i][(j + k) % D])

def psi(x, k, ops: ScalarOpCounter):
    D = len(x[0])
    ops.calls_psi += 1
    # [ψ(x,k)]_{i,j} = x_{(i+k) mod D, j}
    return ops.move_matrix(x, lambda i, j: x[(i + k) % D][j])

# ---- 나이브(기존) 구현 ----
def he_matmul_plain_count_naive(A, B, verbose=False):
    D = len(A)
    ops = ScalarOpCounter()

    SA = sigma(A, ops)    # 1회: D^2 이동
    TB = tau(B, ops)      # 1회: D^2 이동

    C = [[0 for _ in range(D)] for _ in range(D)]

    for k in range(D):
        Xk = pi(SA, k, ops)   # 매 k마다 1회: D^2 이동
        Yk = psi(TB, k, ops)  # 매 k마다 1회: D^2 이동
        # Hadamard + 누적 (스칼라 곱/덧셈 D^2씩)
        Pk = ops.hadamard(Xk, Yk)
        ops.accumulate(C, Pk)

    return C, ops

# ---- BSGS 구현 (m = ceil(sqrt(D)), k = g*m + b) ----
def he_matmul_plain_count_bsgs(A, B, verbose=False):
    D = len(A)
    ops = ScalarOpCounter()

    # Step 1: σ(A), τ(B) (각 1회 재배열)
    SA = sigma(A, ops)
    TB = tau(B, ops)

    # 파라미터 선택: m ~= sqrt(D), n = ceil(D/m)
    m = isqrt(D)
    if m * m < D:
        m += 1
    n = ceil(D / m)   # giant-step 개수

    # Step 2: giant-step만 실제 재배열(이동)로 만들어 캐시
    #   G_A[g] = π(SA, g*m)  for g=0..n-1
    #   G_B[g] = ψ(TB, g*m)  for g=0..n-1
    G_A = []
    G_B = []
    for g in range(n):
        shift = g * m
        G_A.append(pi(SA, shift % D, ops))  # 각 1회: D^2 이동
        G_B.append(psi(TB, shift % D, ops)) # 각 1회: D^2 이동

    # Step 3: 각 k를 g,b로 분해해, baby-step은 "인덱싱"으로만 처리
    #   Xk[i,j] = G_A[g][i][(j+b) mod D]
    #   Yk[i,j] = G_B[g][(i+b) mod D][j]
    #   ==> Xk/Yk를 새로 만들지 않고 Hadamard에서 직접 인덱싱
    C = [[0 for _ in range(D)] for _ in range(D)]
    for k in range(D):
        g = k // m
        b = k - g * m
        GA = G_A[g]
        GB = G_B[g]
        for i in range(D):
            ii = (i + b) % D
            for j in range(D):
                jj = (j + b) % D
                val = GA[i][jj] * GB[ii][j]
                ops.scalar_mults += 1
                C[i][j] += val
                ops.scalar_adds += 1

    return C, ops, (m, n)

# ---- 0/1 대각 B 전용: 나이브 ----
def he_matmul_diag01_naive(A, b_diag, verbose=False, count_trivial_mul: bool = False):
    # HE-스타일 카운트: σ/τ와 k별 π/ψ를 실제로 수행하고, 비영(0/1) 대각의 구조를 이용해
    # Hadamard는 한 행에서만 (#ones)개 곱/덧셈을 수행
    D = len(A)
    assert len(b_diag) == D
    ops = ScalarOpCounter()

    # B를 대각행렬로 구성
    B = [[0 for _ in range(D)] for _ in range(D)]
    for j in range(D):
        B[j][j] = b_diag[j]

    SA = sigma(A, ops)    # D^2 이동
    TB = tau(B, ops)      # D^2 이동 (비제로는 첫 행에만 존재)

    C = [[0 for _ in range(D)] for _ in range(D)]
    ones = sum(1 for v in b_diag if v == 1)

    for k in range(D):
        Xk = pi(SA, k, ops)   # D^2 이동
        Yk = psi(TB, k, ops)  # D^2 이동 → 비제로 행이 i0 = (-k) mod D 로 이동
        i0 = (-k) % D
        # 비제로 열들만 곱/누적
        for j in range(D):
            if b_diag[j] == 1:
                # Yk[i0][j] == 1, 그 외는 0
                if count_trivial_mul:
                    ops.scalar_mults += 1   # 곱하기 1도 곱셈으로 셈
                C[i0][j] += Xk[i0][j]
                ops.scalar_adds += 1

    if verbose:
        print(f"col-ones = {ones} / {D}")

    return C, ops

# ---- 0/1 대각 B 전용: BSGS ----
def he_matmul_diag01_bsgs(A, b_diag, verbose=False, count_trivial_mul: bool = False):
    # HE-스타일 BSGS: σ/τ 후 giant-step만 실제 재배열로 생성, baby-step은 인덱싱으로 처리
    D = len(A)
    assert len(b_diag) == D
    ops = ScalarOpCounter()

    # B를 대각행렬로 구성
    B = [[0 for _ in range(D)] for _ in range(D)]
    for j in range(D):
        B[j][j] = b_diag[j]

    SA = sigma(A, ops)
    TB = tau(B, ops)

    m = isqrt(D)
    if m * m < D:
        m += 1
    n = ceil(D / m)

    # giant steps
    G_A = []
    G_B = []
    for g in range(n):
        shift = g * m
        G_A.append(pi(SA, shift % D, ops))
        G_B.append(psi(TB, shift % D, ops))

    C = [[0 for _ in range(D)] for _ in range(D)]
    ones = sum(1 for v in b_diag if v == 1)

    for k in range(D):
        g = k // m
        b = k - g * m
        GA = G_A[g]
        GB = G_B[g]
        # GB의 비제로 행은 i_g = (-g*m) mod D
        i_g = (-g * m) % D
        # baby-step을 고려한 비제로 행 위치: i0 = (i_g - b) mod D
        i0 = (i_g - b) % D
        for j in range(D):
            if b_diag[j] == 1:
                jj = (j + b) % D
                if count_trivial_mul:
                    ops.scalar_mults += 1   # 곱하기 1도 곱셈으로 셈
                C[i0][j] += GA[i0][jj]
                ops.scalar_adds += 1

    if verbose:
        print(f"m={m}, n={n}, col-ones = {ones} / {D}")

    return C, ops, (m, n)

# ---- 실행 예시 ----
if __name__ == "__main__":
    D = 9
    A = [[D*i + j + 1 for j in range(D)] for i in range(D)]
    # 0/1 대각 B를 벡터로 표현 (열 마스크)
    b = [(j % 2) ^ 1 for j in range(D)]  # 1,0 반복

    # 기준값: A * diag(b)
    C_ref = [[A[i][j] * b[j] for j in range(D)] for i in range(D)]

    print("=== 0/1 대각 B: 나이브 (optimized) ===")
    Cn, on = he_matmul_diag01_naive(A, b, count_trivial_mul=False)
    print("스칼라 곱셈 :", on.scalar_mults)
    print("스칼라 덧셈 :", on.scalar_adds)
    print("원소 이동  :", on.scalar_moves)
    print("calls_pi   :", on.calls_pi)
    print("calls_psi  :", on.calls_psi)
    print()

    print("=== 0/1 대각 B: BSGS (optimized) ===")
    Cb, ob, (m, n) = he_matmul_diag01_bsgs(A, b, count_trivial_mul=False)
    print(f"m = {m}, n = {n}  (D={D} 가정)")
    print("스칼라 곱셈 :", ob.scalar_mults)
    print("스칼라 덧셈 :", ob.scalar_adds)
    print("원소 이동  :", ob.scalar_moves)
    print("calls_pi   :", ob.calls_pi)
    print("calls_psi  :", ob.calls_psi)
    print()

    # 정합성 체크
    assert Cn == C_ref, "diag01 naive 결과가 기준과 다릅니다."
    assert Cb == C_ref, "diag01 BSGS 결과가 기준과 다릅니다."

    print("=== 0/1 대각 B: 나이브 (algebraic count) ===")
    Cn2, on2 = he_matmul_diag01_naive(A, b, count_trivial_mul=True)
    print("스칼라 곱셈 :", on2.scalar_mults, " (기대: D * ones)")
    print("스칼라 덧셈 :", on2.scalar_adds)
    print("원소 이동  :", on2.scalar_moves)
    print()

    print("=== 0/1 대각 B: BSGS (algebraic count) ===")
    Cb2, ob2, _ = he_matmul_diag01_bsgs(A, b, count_trivial_mul=True)
    print("스칼라 곱셈 :", ob2.scalar_mults, " (기대: D * ones)")
    print("스칼라 덧셈 :", ob2.scalar_adds)
    print("원소 이동  :", ob2.scalar_moves)
