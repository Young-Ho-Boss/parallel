from math import sqrt, ceil

# ------------------ 기본 유틸 ------------------

def flatten_matrix(A):  # row-major
    m, n = len(A), len(A[0])
    return [A[i][j] for i in range(m) for j in range(n)]

def unflatten_vector(v, m, n):  # row-major
    return [v[i*n:(i+1)*n] for i in range(m)]

def print_matrix(M):
    for row in M:
        print(" ".join(f"{x:4}" for x in row))

def rotate_vector(vec, k):  # 왼쪽으로 k칸
    n = len(vec); k %= n
    return vec[k:] + vec[:k]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_mul(a, b):  # 성분곱 (CMult로 간주)
    return [a[i] * b[i] for i in range(len(a))]

def shift_mask(mask, s):  # 평문 마스크 인덱스 이동 (비용 0)
    return rotate_vector(mask, s)

# 카운터(참고용)
class Counter:
    def __init__(self):
        self.rot = 0
        self.cmult = 0
        self.add = 0
        self.mult = 0
    def inc_rot(self, k=1):   self.rot += k
    def inc_cmult(self, k=1): self.cmult += k
    def inc_add(self, k=1):   self.add += k
    def inc_mult(self, k=1):  self.mult += k
    def as_tuple(self):       return self.add, self.cmult, self.rot, self.mult
    def __repr__(self):
        return f"Add={self.add}, CMult={self.cmult}, Rot={self.rot}, Mult={self.mult}"

# ------------------ 대각 마스크 생성기 ------------------

# σ: k ∈ {-D+1,...,-1,0,1,...,D-1}
def build_sigma_masks(D: int):
    n = D*D
    masks = {}
    for k in range(-D+1, D):
        m = [0]*n
        if k >= 0:
            # 0 <= ell - D*k < D-k
            base = D*k
            for ell in range(base, base + (D - k)):
                if 0 <= ell < n:
                    m[ell] = 1
        else:
            # -k <= ell - (D+k)*D < D
            base = (D + k) * D
            lo, hi = base + (-k), base + D
            for ell in range(max(0, lo), min(n, hi)):
                m[ell] = 1
        masks[k] = m
    return masks  # dict: k -> mask

# τ: 유효 대각은 d*k (k=0..D-1)
def build_tau_masks(D: int):
    n = D*D
    masks = {}
    for k in range(D):  # k=0..D-1
        m = [0]*n
        for ell in range(k, n, D):  # ell % D == k
            m[ell] = 1
        masks[k] = m   # key는 k (shift는 d*k)
    return masks

# ------------------ BSGS σ / τ ------------------

def bsgs_sigma(vec, D, counter=None):
    """
    입력: vec = vec(A) (길이 D*D)
    출력: vec( sigma(A) )
    BSGS 스케줄:
      k = α*i + j,  α=ceil(sqrt(D)),  -α<i<α,  0<=j<α,  k∈(-D,D)
      baby: ρ(vec; j) 캐시
      s_i = Σ_j [ ρ(vec; j) ⊙ ρ(u_k; -α i) ]
      out += ρ(s_i; α i)
    """
    n = D*D
    α = ceil(sqrt(D))
    masks = build_sigma_masks(D)

    # baby-step rotations
    baby = {}
    for j in range(α):
        if j == 0:
            baby[j] = vec[:]  # 회전 0
        else:
            baby[j] = rotate_vector(vec, j)
            if counter: counter.inc_rot()

    out = None
    first_block = True

    for i in range(-α+1, α):
        s_i = None
        first_term = True
        for j in range(α):
            k = α*i + j
            if k <= -D or k >= D:  # 유효 범위 밖
                continue
            # 마스크 정렬 (평문): u_k shifted by -α i
            mk = masks[k]
            mk_shift = shift_mask(mk, -α*i)  # 비용 0
            term = vec_mul(baby[j], mk_shift)
            if counter: counter.inc_cmult()
            if first_term:
                s_i = term
                first_term = False
            else:
                s_i = vec_add(s_i, term)
                if counter: counter.inc_add()

        if s_i is None:
            continue  # 이 i에 해당하는 유효 j가 없음

        if i != 0:
            s_i = rotate_vector(s_i, α*i)
            if counter: counter.inc_rot()

        if first_block:
            out = s_i
            first_block = False
        else:
            out = vec_add(out, s_i)
            if counter: counter.inc_add()

    return out

def bsgs_tau(vec, D, counter=None):
    """
    입력: vec = vec(B)
    출력: vec( τ(B) )
    BSGS:
      유효 대각 index는 t = 0..D-1, shift는 d*t
      t = α*i + j  분해
      baby: ρ(vec; d*j)
      s_i = Σ_j [ ρ(vec; d*j) ⊙ ρ(u_{d*t}; -dα i) ]
      out += ρ(s_i; dα i)
    """
    n = D*D
    α = ceil(sqrt(D))
    masks = build_tau_masks(D)

    # baby-step rotations (간격 d)
    baby = {}
    for j in range(α):
        sh = D*j
        if j == 0:
            baby[j] = vec[:]  # 회전 0
        else:
            baby[j] = rotate_vector(vec, sh)
            if counter: counter.inc_rot()

    out = None
    first_block = True

    for i in range(-α+1, α):
        s_i = None
        first_term = True
        for j in range(α):
            t = α*i + j  # t ∈ [0, D-1]
            if t < 0 or t >= D:
                continue
            mk = masks[t]                # τ는 key=t, shift는 d*t
            mk_shift = shift_mask(mk, -D*α*i)  # 비용 0
            term = vec_mul(baby[j], mk_shift)
            if counter: counter.inc_cmult()
            if first_term:
                s_i = term
                first_term = False
            else:
                s_i = vec_add(s_i, term)
                if counter: counter.inc_add()

        if s_i is None:
            continue

        if i != 0:
            s_i = rotate_vector(s_i, D*α*i)
            if counter: counter.inc_rot()

        if first_block:
            out = s_i
            first_block = False
        else:
            out = vec_add(out, s_i)
            if counter: counter.inc_add()

    return out

# ------------------ 직관적 σ/τ (행렬 버전) — 검증용 ------------------

def sigma_direct(A):
    D = len(A[0])
    return [[A[i][(i + j) % D] for j in range(D)] for i in range(D)]

def tau_direct(A):
    D = len(A[0])
    return [[A[(i + j) % D][j] for j in range(D)] for i in range(D)]

# ------------------ 테스트 ------------------

if __name__ == "__main__":
    D = 9
    A = [[D*i + j + 1 for j in range(D)] for i in range(D)]
    vA = flatten_matrix(A)

    # σ
    c1 = Counter()
    v_sigma_bsgs = bsgs_sigma(vA, D, counter=c1)
    A_sigma_ref = sigma_direct(A)
    ok_sigma = (unflatten_vector(v_sigma_bsgs, D, D) == A_sigma_ref)

    print("[σ] 값 동일?", ok_sigma)
    print("[σ] 카운트:", c1)  # 회전 수는 스케줄에 따라 이론(3√D)보다 작게 나올 수 있음

    # τ
    c2 = Counter()
    v_tau_bsgs = bsgs_tau(vA, D, counter=c2)
    A_tau_ref = tau_direct(A)
    ok_tau = (unflatten_vector(v_tau_bsgs, D, D) == A_tau_ref)

    print("\n[τ] 값 동일?", ok_tau)
    print("[τ] 카운트:", c2)

    # 보기 좋게 출력
    print("\n== sigma(A) ==")
    print_matrix(unflatten_vector(v_sigma_bsgs, D, D))
    print("\n== tau(A) ==")
    print_matrix(unflatten_vector(v_tau_bsgs, D, D))
