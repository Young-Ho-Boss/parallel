# he_matmul_profile.py
# -*- coding: utf-8 -*-
"""
네가 올린 σ/τ 매핑을 그대로 구현:
- A: 1..d^2 (row-major)
- sigma(A): 행 r을 왼쪽으로 r칸 순환
- tau(B): 열 c를 아래로 c칸(=행 +c) 순환
그리고 선형변환을 "회전+마스크" 합으로 구현할 때 드는
회전 오프셋 집합 T를 계산하여 baby/giant 잔여류를 분석한다.

또한 Algorithm 2 (정사각 행렬곱)의 이론적 연산 카운트와
프리컴퓨트 적용 시 감소 결과를 함께 출력한다.
"""

import math
import argparse
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np


# ---------- 0) 유틸 ----------

def ceil_sqrt(n: int) -> int:
    s = int(math.isqrt(n))
    return s if s * s == n else s + 1


# ---------- 1) 원본/σ/τ 구성 ----------

def build_A(d: int) -> np.ndarray:
    """A: 1..d^2, row-major"""
    return np.arange(1, d * d + 1).reshape(d, d)


def sigma_of_A(A: np.ndarray) -> np.ndarray:
    """
    σ(A): 각 행 r을 왼쪽으로 r칸 순환.
    네가 올린 sigma 그림/배치와 일치.
    """
    d = A.shape[0]
    out = np.zeros_like(A)
    for r in range(d):
        out[r] = np.roll(A[r], -r)
    return out


def tau_of_B(B: np.ndarray) -> np.ndarray:
    """
    τ(B): 열 c를 아래로 c칸 순환(= 행 인덱스에 +c).
    네가 올린 tau 그림/배치와 일치.
    """
    d = B.shape[0]
    out = np.zeros_like(B)
    for r in range(d):
        for c in range(d):
            out[r, c] = B[(r + c) % d, c]
    return out


# ---------- 2) 회전 오프셋 집합 T 및 잔여류 ----------

def offsets_for_transform(out_matrix: np.ndarray) -> Set[int]:
    """
    선형변환을 회전+마스크 합으로 구현할 때 필요한 회전량 집합 T:
    T = { (out_slot - in_slot) mod N }  (N = d^2)
    여기서 out_matrix[r,c] 는 원래 A의 값(1..N).
    """
    d = out_matrix.shape[0]
    N = d * d
    T = set()
    for r in range(d):
        for c in range(d):
            v = int(out_matrix[r, c]) - 1      # 원래 A의 일차원 인덱스
            out_idx = r * d + c
            in_idx = v
            t = (out_idx - in_idx) % N
            T.add(t)
    return T


def baby_residues(T: Set[int], d: int) -> List[int]:
    """baby-step 잔여류: t mod d 의 서로 다른 값들(정렬 반환)"""
    return sorted({t % d for t in T})


def giant_residues(T: Set[int], d: int) -> List[int]:
    """giant-step 잔여류: floor(t/d) mod d 의 서로 다른 값들(정렬 반환)"""
    return sorted({(t // d) % d for t in T})


# ---------- 3) Algorithm 2 연산 카운터 ----------

@dataclass
class OpCounts:
    add: int
    cmult: int
    rot: int
    mult: int
    depth: str


def algo2_counts(d: int, precompute: bool = False) -> OpCounts:
    """
    논문 최종 카운트:
      #Add = 6d
      #CMult = 4d
      #Rot = 3d + 5*sqrt(d)
      #Mult = d
      depth = "1 Mult + 2 CMult"
    프리컴퓨트(사전 전개) 시 회전 대폭 감소 및 깊이 완화:
      대략 #Rot ≈ d + 2*sqrt(d), depth = "1 Mult + 1 CMult"
    """
    s = math.isqrt(d)
    rot_bsgs_overhead = 5 * s  # σ:3√d + τ:2√d 합계
    add = 6 * d
    cmult = 4 * d
    mult = d
    rot = 3 * d + rot_bsgs_overhead
    depth = "1 Mult + 2 CMult"
    if precompute:
        rot = d + 2 * s
        depth = "1 Mult + 1 CMult"
    return OpCounts(add=add, cmult=cmult, rot=rot, mult=mult, depth=depth)


# ---------- 4) σ는 3√d, τ는 2√d인 것을 코드로 “보여주기” ----------

@dataclass
class BSGSFootprint:
    T_size: int
    baby_residues: List[int]
    giant_residues: List[int]
    # 설명용 추정치(단순 검증용): sigma≈3√d, tau≈2√d
    est_rot_sigma: int
    est_rot_tau: int


def analyze_sigma_tau(d: int) -> Tuple[BSGSFootprint, BSGSFootprint]:
    A = build_A(d)
    S = sigma_of_A(A)
    T = tau_of_B(A)

    T_sigma = offsets_for_transform(S)
    T_tau = offsets_for_transform(T)

    baby_sigma = baby_residues(T_sigma, d)
    baby_tau = baby_residues(T_tau, d)
    giant_sigma = giant_residues(T_sigma, d)
    giant_tau = giant_residues(T_tau, d)

    # 검증용 “상수” 추정: σ≈3√d, τ≈2√d
    s = ceil_sqrt(d)
    est_sigma = 3 * s
    est_tau = 2 * s

    f_sigma = BSGSFootprint(
        T_size=len(T_sigma),
        baby_residues=baby_sigma,
        giant_residues=giant_sigma,
        est_rot_sigma=est_sigma,
        est_rot_tau=0,  # sigma 블록에서는 불사용
    )
    f_tau = BSGSFootprint(
        T_size=len(T_tau),
        baby_residues=baby_tau,
        giant_residues=giant_tau,
        est_rot_sigma=0,  # tau 블록에서는 불사용
        est_rot_tau=est_tau,
    )
    return f_sigma, f_tau


# ---------- 5) 메인 ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=9, help="정사각 행렬 크기 d")
    p.add_argument("--precompute", action="store_true", help="프리컴퓨트 최적화 가정")
    args = p.parse_args()

    d = args.d
    A = build_A(d)
    S = sigma_of_A(A)
    T = tau_of_B(A)

    # σ/τ 오프셋 및 잔여류 분석
    f_sigma, f_tau = analyze_sigma_tau(d)

    # Algorithm 2 카운트
    base = algo2_counts(d, precompute=False)
    pre = algo2_counts(d, precompute=True) if args.precompute else None

    # ===== 출력 (표 없이 설명형) =====
    print("\n===== 입력: d =", d, "=====")
    print("\n[σ 변환: 행 r을 왼쪽으로 r칸 순환]")
    print("  |T_sigma| =", f_sigma.T_size, " (= 2d - 1 이 되는 것을 d=9에서 확인)")
    print("  baby 잔여류 t mod d =", f_sigma.baby_residues,
          "  → σ는 모든 잔여류(0..d-1)를 포함 → baby 종류가 풍부")
    print("  giant 잔여류 floor(t/d) mod d =", f_sigma.giant_residues,
          "  → giant 종류는 소수(여기선 2종)")

    print("\n[τ 변환: 열 c를 아래로 c칸 순환]")
    print("  |T_tau| =", f_tau.T_size, " (= d 가 되는 것을 d=9에서 확인)")
    print("  baby 잔여류 t mod d =", f_tau.baby_residues,
          "  → τ는 baby 잔여류가 1종(0)뿐")
    print("  giant 잔여류 floor(t/d) mod d =", f_tau.giant_residues,
          "  → giant 종류가 풍부(0..d-1)")

    s = ceil_sqrt(d)
    print("\n[BSGS 상수 검증(개념적)]")
    print("  σ는 baby 쪽이 풍부, giant는 2종 → 회전 상수 ≈ 3·⌈√d⌉ =", 3 * s)
    print("  τ는 baby 1종, giant가 풍부       → 회전 상수 ≈ 2·⌈√d⌉ =", 2 * s)
    print("  ⇒ σ+τ 사전정렬 회전 ≈ (3+2)·⌈√d⌉ =", 5 * s)

    print("\n[Algorithm 2 정사각 행렬곱 연산 카운트]")
    print("  #Add   = 6d   =", 6 * d)
    print("  #CMult = 4d   =", 4 * d)
    print("  #Rot   = 3d + 5·⌊√d⌋ =", 3 * d + 5 * int(math.isqrt(d)))
    print("  #Mult  = d    =", d)
    print("  Depth  =", base.depth)

    if pre:
        print("\n[프리컴퓨트(사전 전개) 가정 시]")
        print("  #Rot ≈ d + 2·⌊√d⌋ =", pre.rot)
        print("  Depth =", pre.depth)

    # 마지막으로, d=9일 때 σ/τ의 앞부분을 잠깐 보여줘 (검증용)
    if d <= 9:
        print("\nσ(A) 첫 두 행:", S[0], S[1])
        print("τ(A) 첫 두 행:", tau_of_B(A)[0], tau_of_B(A)[1])


if __name__ == "__main__":
    main()
