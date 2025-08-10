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

# 벡터 회전 (좌측으로 k칸, 모듈러)
def rotate_vector(A: list, k: int) -> list:
    n = len(A)
    k = k % n
    return [A[(i + k) % n] for i in range(n)]

# 대각벡터 추출: U (n x n) -> [u_0, u_1, ..., u_{n-1}]
# u_l[t] = U[t, (t+l) mod n]
def diagonal_vector(U: list) -> list:
    n = len(U)
    dVecs = []
    for i in range(n):
        dVec = []
        for j in range(n):
            dVec.append(U[j][(j+i) % n])
        dVecs.append(dVec)
    return dVecs

# 테스트용 - 일반 행렬 곱셈
def mult_matrix(A: list, B: list) -> list:
    if isinstance(B[0], (int, float)):
        B = [[x] for x in B]
    assert len(A[0]) == len(B)
    rowA, colA, colB = len(A), len(A[0]), len(B[0])
    result = [[0 for _ in range(colB)] for _ in range(rowA)]
    for i in range(rowA):
        for j in range(colB):
            s = 0
            for k in range(colA):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

# 성분별 연산
def mult_vector(A: list, B: list) -> list:
    return [A[i]*B[i] for i in range(len(A))]

def add_vector(A: list, B: list) -> list:
    return [A[i]+B[i] for i in range(len(A))]

# 행렬 복제(수직 방향으로 l행 블록을 D까지 채워 d x d 행렬을 만들기)
def duplicate_matrix(A: list, d: int, l: int) -> list:
    assert d % l == 0
    count = d // l
    barA = []
    for _ in range(count):
        for row in A:
            barA.append(row)
    return barA

# 보기용 출력
def print_matrix(M: list, mode: str = "debug", step_name: str = None):
    rows = len(M)
    cols = len(M[0])
    if step_name:
        print(f"\n{step_name} (size {rows}x{cols}):")
    else:
        print(f"\nMatrix (size {rows}x{cols}):")
    print("-" * (cols * 8))
    for i in range(rows):
        row = [str(M[i][j]).rjust(6) for j in range(cols)]
        print(" ".join(row))

def print_vector(A: list, mode="debug", step_name: str = None) -> None:
    if step_name:
        print(f"\n{step_name} (size {len(A)}):")
    else:
        print(f"\nVector (size {len(A)}):")
    row = [str(A[i]).rjust(6) for i in range(len(A))]
    print(" ".join(row))

def print_vector_matrixstyle(A: list, d: int, mode="debug", step_name: str = None):
    cols = len(A) // d
    matrixA = [A[i * cols:(i + 1) * cols] for i in range(d)]
    print_matrix(matrixA, mode, step_name)

# =========================================
# 순열 행렬 생성 (U^σ, U^τ, U^{φ^k}, U^{ψ^k})
# =========================================

def sigma_Matrix(d: int) -> list:
    size = d ** 2
    U = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * i + ((i + j) % d)
            U[row][col] = 1
    return U

def tau_Matrix(d: int) -> list:
    size = d ** 2
    U = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * ((i + j) % d) + j
            U[row][col] = 1
    return U

def phi_Matrix(d: int, k: int) -> list:
    size = d ** 2
    U = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * i + ((j + k) % d)
            U[row][col] = 1
    return U

def psi_Matrix(d: int, k: int) -> list:
    size = d ** 2
    U = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(d):
        for j in range(d):
            row = d * i + j
            col = d * ((i + k) % d) + j
            U[row][col] = 1
    return U

# =========================================
# HE 연산 카운터
# =========================================

def ceil_sqrt(n: int) -> int:
    r = int(sqrt(n))
    return r if r*r == n else r+1

class HEOpCounter:
    def __init__(self, name=""):
        self.name = name
        self.Add = 0
        self.CMult = 0
        self.Rot = 0
        self.Mult = 0
    def inc_add(self, k=1): self.Add += k
    def inc_cmult(self, k=1): self.CMult += k
    def inc_rot(self, k=1): self.Rot += k
    def inc_mult(self, k=1): self.Mult += k
    def snapshot(self):
        return {"Add": self.Add, "CMult": self.CMult, "Rot": self.Rot, "Mult": self.Mult}
    def reset(self):
        self.Add = self.CMult = self.Rot = self.Mult = 0

def pretty_counts(title, cdict):
    print(f"\n[{title}]")
    print(f"Add   = {cdict['Add']}")
    print(f"CMult = {cdict['CMult']}")
    print(f"Rot   = {cdict['Rot']}")
    print(f"Mult  = {cdict['Mult']}")

# =========================================
# 선형변환 (계측 버전)
# U의 대각분해를 이용해 회전/상수곱/덧셈 카운팅
# =========================================

def lintrans_counted(ct: list, U: list, counter: HEOpCounter) -> list:
    n = len(ct)
    dVecs = diagonal_vector(U)
    acc = [0]*n
    first = True
    for shift, mask in enumerate(dVecs):
        if not any(mask):    # 0 벡터면 스킵
            continue
        if shift % n != 0:   # shift=0은 회전 없음
            counter.inc_rot(1)
        counter.inc_cmult(1) # 상수곱
        rotated = rotate_vector(ct, shift)
        prod = mult_vector(rotated, mask)
        if first:
            acc = prod[:]
            first = False
        else:
            counter.inc_add(1)
            acc = add_vector(acc, prod)
    return acc

def hadamard_mult_counted(x, y, counter: HEOpCounter):
    counter.inc_mult(1)
    return mult_vector(x, y)

def hadamard_add_counted(x, y, counter: HEOpCounter):
    counter.inc_add(1)
    return add_vector(x, y)

# =========================================
# Algorithm 2 (정사각) - 계측
# =========================================

def algol2_counted(A, B, mode="debug"):
    d = len(A)
    counter = HEOpCounter("ALG2")

    # Step 1-1: σ
    ctA = flatten_matrix(A)
    A0 = lintrans_counted(ctA, sigma_Matrix(d), counter)
    if mode == "debug":
        print_vector_matrixstyle(A0, d, "debug", "ct.A0 (σ)")

    # Step 1-2: τ
    ctB = flatten_matrix(B)
    B0 = lintrans_counted(ctB, tau_Matrix(d), counter)
    if mode == "debug":
        print_vector_matrixstyle(B0, d, "debug", "ct.B0 (τ)")

    # Step 2: φ^k, ψ^k (k=1..d-1)
    ctAk = [A0]
    ctBk = [B0]
    for k in range(1, d):
        ctAk.append(lintrans_counted(A0, phi_Matrix(d, k), counter))
        ctBk.append(lintrans_counted(B0, psi_Matrix(d, k), counter))
        if mode == "debug":
            print_vector_matrixstyle(ctAk[k], d, "debug", f"ct.A{k} (φ^{k})")
            print_vector_matrixstyle(ctBk[k], d, "debug", f"ct.B{k} (ψ^{k})")

    # Step 3: Hadamard products & sum
    ctAB = hadamard_mult_counted(A0, B0, counter)
    for k in range(1, d):
        term = hadamard_mult_counted(ctAk[k], ctBk[k], counter)
        ctAB = hadamard_add_counted(ctAB, term, counter)

    if mode == "debug":
        print_vector_matrixstyle(ctAB, d, "debug", "Algorithm 2 (result)")

    return ctAB, counter.snapshot()

# =========================================
# Algorithm 3 (직사각) - 계측
# R_{ℓ×D} × R_{D×D} → R_{ℓ×D}
# =========================================

def algol3_counted(A, B, mode="debug"):
    ell = len(A)
    D = len(A[0])
    assert D % ell == 0, "ℓ은 D의 약수여야 합니다."
    counter = HEOpCounter("ALG3")

    # Step 0: barA
    barA = duplicate_matrix(A, D, ell)

    # Step 1: σ, τ
    ctA = flatten_matrix(barA)
    A0 = lintrans_counted(ctA, sigma_Matrix(D), counter)
    ctB = flatten_matrix(B)
    B0 = lintrans_counted(ctB, tau_Matrix(D), counter)
    if mode == "debug":
        print_vector_matrixstyle(A0, D, "debug", "ct.A0 (σ)")
        print_vector_matrixstyle(B0, D, "debug", "ct.B0 (τ)")

    # Step 2: φ^i, ψ^i (i=1..ell-1)
    ctAk = [A0]
    ctBk = [B0]
    for i in range(1, ell):
        ctAk.append(lintrans_counted(A0, phi_Matrix(D, i), counter))
        ctBk.append(lintrans_counted(B0, psi_Matrix(D, i), counter))
        if mode == "debug":
            print_vector_matrixstyle(ctAk[i], D, "debug", f"ct.A{i} (φ^{i})")
            print_vector_matrixstyle(ctBk[i], D, "debug", f"ct.B{i} (ψ^{i})")

    # Step 3: Hadamard ell회
    ctbarAB = hadamard_mult_counted(A0, B0, counter)
    for i in range(1, ell):
        term = hadamard_mult_counted(ctAk[i], ctBk[i], counter)
        ctbarAB = hadamard_add_counted(ctbarAB, term, counter)

    # Step 4: 블록 합산(반복-두배) → ceil(log2(D/ell))번 회전+덧셈
    r = D // ell
    t = ceil(log2(r)) if r > 1 else 0
    ctAB = ctbarAB[:]
    for k in range(t):
        counter.inc_rot(1)
        rotated = rotate_vector(ctAB, ell * D * (2**k))
        counter.inc_add(1)
        ctAB = add_vector(ctAB, rotated)

    if mode == "debug":
        print_vector_matrixstyle(ctAB, D, "debug", "Algorithm 3 (result)")

    return ctAB, counter.snapshot()

# =========================================
# 이론(최적화, BSGS+MUX) 합계식
# =========================================

def theory_square_counts(D: int):
    # Algorithm 2: Add=6D, CMult=4D, Rot=3D+5√D, Mult=D
    return {
        "Add": 6*D,
        "CMult": 4*D,
        "Rot": 3*D + 5*ceil_sqrt(D),
        "Mult": D
    }

def theory_rect_counts(D: int, ell: int):
    # Algorithm 3: Add=3D+2ℓ+ceil(log2(D/ℓ)), CMult=3D+2ℓ,
    #              Rot=3ℓ+5√D+ceil(log2(D/ℓ)), Mult=ℓ
    r = D // ell
    lg = ceil(log2(r)) if r > 1 else 0
    return {
        "Add": 3*D + 2*ell + lg,
        "CMult": 3*D + 2*ell,
        "Rot": 3*ell + 5*ceil_sqrt(D) + lg,
        "Mult": ell
    }

def pretty_compare_square(D, measured):
    theo = theory_square_counts(D)
    pretty_counts(f"Algorithm 2 (정사각, D={D}) — 계측 결과", measured)
    pretty_counts(f"Algorithm 2 (정사각, D={D}) — 이론(최적화)", theo)

def pretty_compare_rect(D, ell, measured):
    theo = theory_rect_counts(D, ell)
    pretty_counts(f"Algorithm 3 (직사각, ℓ={ell}, D={D}) — 계측 결과", measured)
    pretty_counts(f"Algorithm 3 (직사각, ℓ={ell}, D={D}) — 이론(최적화)", theo)

# =========================================
# 데모 (D=9, ℓ=3)
# =========================================

if __name__ == "__main__":
    # 정사각 D=9
    D = 9
    A = [[i*D + j + 1 for j in range(D)] for i in range(D)]
    # B는 단위행렬(검증 용이)
    B = [[1 if i==j else 0 for j in range(D)] for i in range(D)]
    _, m_sq = algol2_counted(A, B, mode="debug")
    pretty_compare_square(D, m_sq)

    # 직사각 ℓ=3, D=9  (3x9) x (9x9)
    ell = 3
    A_rect = [[i*D + j + 1 for j in range(D)] for i in range(ell)]
    B_rect = [[1 if i==j else 0 for j in range(D)] for i in range(D)]
    _, m_rc = algol3_counted(A_rect, B_rect, mode="debug")
    pretty_compare_rect(D, ell, m_rc)
