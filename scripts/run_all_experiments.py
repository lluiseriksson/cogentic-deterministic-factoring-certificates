"""Generate all 16 canonical exact certificates and 4 exact numerical datasets for Paper 61:
Unconditional Deterministic Subexponential Integer Factoring and Quadratic Nonresidue
Construction (Cogentic 2026: O(exp(3 (ln N)^(3/4))) & L_p[1/2, O(1)]).
"""

from __future__ import annotations

import json
import sys
import time
from math import exp, gcd, log
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detfact_certifier import SCHEMA_VERSION, verify_certificate_dict
from src.detfact_exact import (
    approx_log_exact,
    approx_sqrt_exact,
    exact_det_and_adjugate,
    first_k_primes,
    gauss_reduction_cycle_real,
    is_irreducible_f2,
    is_prime_exact,
    lcm_1_to_k,
    lcm_int,
    legendre_symbol,
    miller_multiquadratic_roots,
    mod_hnf_exact,
    mod_inv,
    poly_eval_int,
    poly_eval_mod,
    walsh_hadamard_norm_sums_exact,
)

CERT_DIR = ROOT / "certificates"
RES_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers for Mode 1, Mode 2, Mode 3, Mode 4 Certificate Construction
# ---------------------------------------------------------------------------


def make_mode1_cert(cert_id: str, description: str, N: int, M_tilde: int, K: int) -> Dict[str, Any]:
    res = miller_multiquadratic_roots(N, M_tilde, K)
    payload: Dict[str, Any] = {
        "N": N,
        "M_tilde": M_tilde,
        "K": K,
        "claimed_outcome": res["outcome"],
    }
    if res["outcome"] == "SPLIT":
        payload["claimed_divisor"] = res["divisor"]
    else:
        payload["claimed_j_star"] = res["j_star"]
        payload["claimed_t_star"] = res["t_star"]
        payload["claimed_family"] = res["family"]
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "miller_multiquadratic_compat_cert",
        "description": description,
        "payload": payload,
    }


def make_mode2_cert(
    cert_id: str,
    description: str,
    q_prime: int,
    K: int,
    X: int,
    Y: int,
    gauss_real_d: int,
) -> Dict[str, Any]:
    res_m = miller_multiquadratic_roots(q_prime, q_prime - 1, K)
    assert res_m["outcome"] == "COMPATIBLE_FAMILY"
    fam = res_m["family"]
    gres = gauss_reduction_cycle_real(gauss_real_d)
    wh = walsh_hadamard_norm_sums_exact(q_prime, fam, K, X, Y)
    assert wh["chi_on_Y_all_plus"] and wh["all_wh_nonneg"] and wh["all_smooth_ge_1"] and wh["lemma51_3_holds"]
    # Find least quadratic nonresidue z mod q_prime
    qnr_z = next(z for z in range(2, q_prime) if legendre_symbol(z, q_prime) == -1)
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "quadratic_infrastructure_wh_cert",
        "description": description,
        "payload": {
            "q_prime": q_prime,
            "K": K,
            "X": X,
            "Y": Y,
            "family": fam,
            "gauss_real_d": gauss_real_d,
            "claimed_gauss_period": gres["period_L"],
            "claimed_unit_x": gres["unit_x"],
            "claimed_unit_y": gres["unit_y"],
            "claimed_unit_norm": gres["unit_norm"],
            "claimed_smooth_Psi": wh["smooth_count_Psi"],
            "claimed_max_S_ell": wh["max_S_ell"],
            "claimed_qnr_z": qnr_z,
        },
    }


def make_mode3_cert(
    cert_id: str,
    description: str,
    al_params: Tuple[int, int, int],
    asq_params: Tuple[int, int, int],
    hnf_m: int,
    hnf_dmod: int,
    hnf_cols: List[List[int]],
    N_tp: int,
    ell_tp: int,
    B_mat: List[List[int]],
    Q_lcm_hint: int,
) -> Dict[str, Any]:
    u_l, v_l, M_l = al_params
    num_l, _ = approx_log_exact(u_l, v_l, M_l)
    u_s, v_s, M_s = asq_params
    num_s, _ = approx_sqrt_exact(u_s, v_s, M_s)
    H_mat = mod_hnf_exact(hnf_cols, hnf_m, hnf_dmod)

    det_B, adj_B = exact_det_and_adjugate(B_mat)
    E0 = abs(det_B)
    # Find exact minimal Q_lcm multiple of Q_lcm_hint so ell^{E0 * Q_lcm} == +-1 (mod N_tp)
    # Specifically compute order of ell^{E0} mod N_tp (small N_tp)
    base = pow(ell_tp, E0, N_tp)
    ord_half = 1
    cur = base
    while cur not in (1, N_tp - 1):
        cur = (cur * base) % N_tp
        ord_half += 1
    Q_lcm = lcm_int(Q_lcm_hint, ord_half)
    val_half = pow(ell_tp, E0 * Q_lcm, N_tp)
    # Pick s2 = -1 (or +1) and P_mod = s2 mod N_tp; then ell^{E0*Q_lcm} * P_mod == s1 (mod N_tp)
    s2 = -1
    P_mod = (-1) % N_tp
    s1 = -1 if val_half == 1 else 1
    assert (val_half * P_mod - s1) % N_tp == 0
    E_ell = 2 * E0 * Q_lcm
    assert pow(ell_tp, E_ell, N_tp) == 1

    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "fixedpoint_hnf_and_trypair_cert",
        "description": description,
        "payload": {
            "approx_log_probe": {"u": u_l, "v": v_l, "M": M_l, "claimed_num": num_l},
            "approx_sqrt_probe": {"u": u_s, "v": v_s, "M": M_s, "claimed_num": num_s},
            "mod_hnf_probe": {"m": hnf_m, "d_mod": hnf_dmod, "columns": hnf_cols, "claimed_H": H_mat},
            "trypair_annihilation_probe": {
                "N": N_tp,
                "ell": ell_tp,
                "pivot_matrix_B": B_mat,
                "claimed_det_B": det_B,
                "claimed_adj_B": adj_B,
                "E0": E0,
                "Q_lcm": Q_lcm,
                "P_mod_N": P_mod,
                "s1": s1,
                "s2": s2,
                "claimed_E_ell": E_ell,
            },
        },
    }


def _poly_mul_mod(a: List[int], b: List[int], mod: int) -> List[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            out[i + j] = (out[i + j] + ca * cb) % mod
    return out


def build_lemma81_poly_pair(
    N: int,
    d: int,
    z0: int,
    m: int,
    ell: int,
) -> Dict[str, Any]:
    """Exact deterministic implementation of Cogentic Lemma 8.1 preconditioned base-m pair."""
    P0 = 1
    for p in range(2, z0 + 1):
        if is_prime_exact(p):
            P0 *= p
    Podd = P0 // 2
    assert gcd(N, P0 * ell) == 1
    assert (m - 1) ** d < N < m ** d

    # Choose odd cd >= 4 * P0 * ell divisible by Podd with supp(cd) subseteq [3, z0]
    cd = Podd
    while cd < 4 * P0 * ell:
        cd *= 3

    # Lexicographically first monic irreducible h2 in F_2[X] of degree d
    h2: List[int] = []
    for mask in range(1 << d):
        cand = [(mask >> j) & 1 for j in range(d)] + [1]
        if is_irreducible_f2(cand):
            h2 = cand
            break
    assert len(h2) == d + 1

    # h1_ell(X) = cd * prod_{s=1}^d (X - r_s) mod ell where r_s are d smallest in {0..ell-1} \ {m mod ell}
    m_mod_ell = m % ell
    roots = [r for r in range(ell) if r != m_mod_ell][:d]
    h1_ell = [cd % ell]
    for r in roots:
        h1_ell = _poly_mul_mod(h1_ell, [(-r) % ell, 1], ell)

    # h2_ell(X) = cd * (X^d - X - s_ell) mod ell where s_ell absent from {x^d - x mod ell}
    img_vals = {(pow(x, d, ell) - x) % ell for x in range(ell)}
    s_ell = next(s for s in range(ell) if s not in img_vals)
    h2_ell = [0] * (d + 1)
    h2_ell[0] = (-cd * s_ell) % ell
    h2_ell[1] = (-cd) % ell
    h2_ell[d] = cd % ell

    # Exact CRT lift for coefficients j = 0..d-1 modulo M_crt = P0 * ell = 2 * Podd * ell
    M_crt = P0 * ell

    def crt_coeff(r2: int, r_odd: int, r_ell: int) -> int:
        # Pairwise coprime moduli: 2, Podd, ell
        for x in range(M_crt):
            if x % 2 == r2 and x % Podd == r_odd and x % ell == r_ell:
                return x
        raise RuntimeError("CRT failed")

    # Build f_i(X) = cd * X^d + sum_{j=0}^{d-1} c_j X^j satisfying:
    #   c_j == h2[j] (mod 2), c_j == (1 if j==0 else 0) (mod Podd), c_j == h_{i,ell}[j] (mod ell)
    #   AND f_i(m) == 0 (mod N) with 1 <= f_i(m)/N < 2*cd and gcd(f_i(m)/N, P0*ell) == 1!
    # Notice adjusting c_0 by a multiple of M_crt preserves ALL congruences mod 2, Podd, ell!
    # Pick c_j = crt_coeff(...) for j = 1..d-1, and solve for c_0 == a_0 (mod M_crt) such that
    # f_i(m) == 0 (mod N), i.e., c_0 = a_0 + M_crt * t with base_eval + M_crt * t == 0 (mod N)!
    # Since gcd(M_crt, N) = 1, unique t0 in [0, N-1] makes f_i(m) divisible by N!
    # Check if k0 = f_i(m)/N is in [1, 2*cd) and coprime to M_crt; if not, shift c_1 by +M_crt * step!
    inv_Mcrt = mod_inv(M_crt % N, N)

    def build_one(h_ell: List[int]) -> Tuple[List[int], int]:
        base_a = [
            crt_coeff(h2[j], 1 if j == 0 else 0, h_ell[j])
            for j in range(d)
        ]
        for shift_c1 in range(0, 200):
            coeffs = list(base_a) + [cd]
            coeffs[1] += shift_c1 * M_crt
            val0 = poly_eval_int(coeffs, m)
            # We want val0 + M_crt * t == 0 (mod N) with t possibly negative or positive near 0
            t_mod = ((-val0) % N) * inv_Mcrt % N
            for t in (t_mod, t_mod - N):
                c_try = list(coeffs)
                c_try[0] += t * M_crt
                vm = poly_eval_int(c_try, m)
                if vm % N == 0:
                    k0 = vm // N
                    if 1 <= k0 < 2 * cd and gcd(k0, M_crt) == 1:
                        # Double check f(a) == 1 (mod P0) for all a in [0, P0-1]:
                        # Note f(a) mod 2 = h2(a) mod 2 (Wait: irreducible deg d >= 4 over F_2 has NO roots in F_2,
                        # so h2(0)=1 and h2(1)=1 automatically! Hence f(a) == 1 (mod 2) AND f(a) == 1 (mod Podd) => f(a) == 1 (mod P0)!)
                        if all(poly_eval_mod(c_try, a, P0) == 1 for a in range(P0)):
                            return c_try, k0
        raise RuntimeError("Failed to find Lemma 8.1 polynomial")

    f1, k0 = build_one(h1_ell)
    f2, k0_p = build_one(h2_ell)
    return {
        "d": d,
        "z0": z0,
        "m": m,
        "ell": ell,
        "f1_coeffs": f1,
        "f2_coeffs": f2,
        "claimed_k0": k0,
        "claimed_k0_prime": k0_p,
    }


def make_mode4_cert(
    cert_id: str,
    description: str,
    p1: int,
    p2: int,
    d: int,
    z0: int,
    ell: int,
    P_ok: List[int],
    K_idx: int,
) -> Dict[str, Any]:
    N = p1 * p2
    # m = ceil(N^(1/d)) so (m-1)^d < N < m^d
    m = int(round(N ** (1.0 / d)))
    while m ** d <= N:
        m += 1
    while (m - 1) ** d >= N:
        m -= 1
    assert (m - 1) ** d < N < m ** d

    bm = build_lemma81_poly_pair(N, d, z0, m, ell)

    # Compute exact multiplicative order of subgroup <P_ok> mod p1 and mod p2!
    def mult_order_mod_p(a: int, p: int) -> int:
        phi = p - 1
        divs = [x for x in range(1, phi + 1) if phi % x == 0]
        for dv in divs:
            if pow(a, dv, p) == 1:
                return dv
        return phi

    ord1 = 1
    ord2 = 1
    for q in P_ok:
        ord1 = lcm_int(ord1, mult_order_mod_p(q, p1))
        ord2 = lcm_int(ord2, mult_order_mod_p(q, p2))
    M_star = lcm_int(ord1, ord2)
    if M_star % 2 != 0:
        M_star *= 2
    L_idx = lcm_1_to_k(K_idx)
    M_tilde = M_star * L_idx
    assert M_tilde % (p1 - 1) == 0 and M_tilde % (p2 - 1) == 0

    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "base_m_saturation_and_factor_cert",
        "description": description,
        "payload": {
            "N": N,
            "factor_p1": p1,
            "factor_p2": p2,
            "base_m_pair": bm,
            "saturation_probe": {
                "P_ok": P_ok,
                "M_star": M_star,
                "K_idx": K_idx,
                "claimed_L_idx": L_idx,
                "claimed_M_tilde": M_tilde,
            },
        },
    }


def build_all_16_certificates() -> List[Dict[str, Any]]:
    certs: List[Dict[str, Any]] = []

    # ---- MODE 1: 4 Canonical Certificates (Miller 2-Sylow Split & Multiquadratic Compatibility) ----
    # 1. Immediate Miller 2-Sylow split on N = 17 * 19 = 323
    certs.append(
        make_mode1_cert(
            "m1_miller_split_n323",
            "Sec. 3 Miller 2-Sylow order test deterministically splits N = 17 * 19 = 323 given M_tilde = 144",
            N=323,
            M_tilde=144,
            K=3,
        )
    )
    # 2. Immediate Miller 2-Sylow split on N = 101 * 103 = 10403
    certs.append(
        make_mode1_cert(
            "m1_miller_split_n10403",
            "Sec. 3 Miller 2-Sylow order test deterministically splits N = 101 * 103 = 10403 given M_tilde = 5100",
            N=10403,
            M_tilde=5100,
            K=3,
        )
    )
    # 3. Prime p = 211 with K = 3 -> 7 compatible quadratic roots & 21 biquadratic compatibility pairs
    certs.append(
        make_mode1_cert(
            "m1_compat_family_p211_k3",
            "Prime p = 211, K = 3: exact 7-field compatible multiquadratic roots & 21 biquadratic identities",
            N=211,
            M_tilde=210,
            K=3,
        )
    )
    # 4. Find smallest semiprime N = p1 * p2 (p1 < p2) that yields COMPATIBLE_FAMILY at K = 4 (15 fields, 105 pairs!)
    # That happens when all q in {-1, 2, 3, 5, 7} have matching 2-adic characters on p1 and p2 (e.g. p1 = 311, p2 = 479,
    # both == 7 mod 8 with (2/p)=(3/p)=(5/p)=(7/p)=+1 and (-1/p)=-1!).
    # Check 311 * 479 = 148,969!
    M_semi4 = lcm_int(310, 478)
    certs.append(
        make_mode1_cert(
            "m1_compat_family_semi148969_k4",
            "Semiprime N = 311 * 479 = 148969, K = 4: 15 compatible multiquadratic roots & 105 biquadratic identities",
            N=311 * 479,
            M_tilde=M_semi4,
            K=4,
        )
    )

    # ---- MODE 2: 4 Canonical Certificates (Gauss Reduction Cycles + Walsh-Hadamard Positivity + QNR) ----
    # 5. q = 71 (least QNR z = 7, chi=+1 on [1..6]), K=2, X=120, Y=6, real Gauss d=2 (D=8, period 2, norm -1)
    certs.append(
        make_mode2_cert(
            "m2_wh_q71_gauss_d2",
            "Prime q=71 (QNR z=7), K=2, Y=6, X=120: Walsh-Hadamard positivity + Gauss cycle d=2 (D=8)",
            q_prime=71,
            K=2,
            X=120,
            Y=6,
            gauss_real_d=2,
        )
    )
    # 6. q = 311 (least QNR z = 11, chi=+1 on [1..10]), K=3, X=250, Y=10, real Gauss d=13 (D=13, period 5, norm -1)
    certs.append(
        make_mode2_cert(
            "m2_wh_q311_gauss_d13",
            "Prime q=311 (QNR z=11), K=3, Y=10, X=250: 7-field Walsh-Hadamard positivity + Gauss cycle d=13",
            q_prime=311,
            K=3,
            X=250,
            Y=10,
            gauss_real_d=13,
        )
    )
    # 7. q = 479 (least QNR z = 13, chi=+1 on [1..12]), K=3, X=400, Y=12, real Gauss d=29 (D=29, period 5, norm -1)
    certs.append(
        make_mode2_cert(
            "m2_wh_q479_gauss_d29",
            "Prime q=479 (QNR z=13), K=3, Y=12, X=400: 7-field Walsh-Hadamard positivity + Gauss cycle d=29",
            q_prime=479,
            K=3,
            X=400,
            Y=12,
            gauss_real_d=29,
        )
    )
    # 8. q = 1559 (least QNR z = 17, chi=+1 on [1..16]), K=4, X=600, Y=16, real Gauss d=43 (D=172, period 10, norm +1)
    certs.append(
        make_mode2_cert(
            "m2_wh_q1559_gauss_d43",
            "Prime q=1559 (QNR z=17), K=4, Y=16, X=600: 15-field Walsh-Hadamard positivity + Gauss cycle d=43 (L=10)",
            q_prime=1559,
            K=4,
            X=600,
            Y=16,
            gauss_real_d=43,
        )
    )

    # ---- MODE 3: 4 Canonical Certificates (ApproxLog/Sqrt, ModHNF, Bordered-Adjugate & TryPair) ----
    # 9. m=2 ModHNF + 2x2 bordered adjugate + TryPair mod N=323 (ell=7)
    certs.append(
        make_mode3_cert(
            "m3_fp_hnf2_trypair_n323",
            "Lemma 2.2 ApproxLog/Sqrt (M=32) + 2D ModHNF + 2x2 TryPair localization annihilation mod N=323",
            al_params=(17, 5, 32),
            asq_params=(13, 3, 32),
            hnf_m=2,
            hnf_dmod=60,
            hnf_cols=[[12, 18], [30, 45], [6, 9]],
            N_tp=323,
            ell_tp=7,
            B_mat=[[3, 1], [2, 5]],
            Q_lcm_hint=2,
        )
    )
    # 10. m=3 ModHNF + 3x3 bordered adjugate + TryPair mod N=899 (29*31, ell=11)
    certs.append(
        make_mode3_cert(
            "m3_fp_hnf3_trypair_n899",
            "Lemma 2.2 ApproxLog/Sqrt (M=48) + 3D ModHNF + 3x3 TryPair localization annihilation mod N=899",
            al_params=(257, 13, 48),
            asq_params=(43, 7, 48),
            hnf_m=3,
            hnf_dmod=120,
            hnf_cols=[[10, 20, 30], [15, 25, 45], [6, 12, 18], [40, 80, 100]],
            N_tp=899,
            ell_tp=11,
            B_mat=[[2, 1, 0], [1, 3, 1], [0, 1, 4]],
            Q_lcm_hint=3,
        )
    )
    # 11. m=4 ModHNF + 4x4 bordered adjugate + TryPair mod N=1763 (41*43, ell=13)
    certs.append(
        make_mode3_cert(
            "m3_fp_hnf4_trypair_n1763",
            "Lemma 2.2 ApproxLog/Sqrt (M=64) + 4D biquadratic ModHNF + 4x4 TryPair annihilation mod N=1763",
            al_params=(1025, 19, 64),
            asq_params=(172, 11, 64),
            hnf_m=4,
            hnf_dmod=84,
            hnf_cols=[[14, 28, 42, 56], [21, 7, 35, 14], [12, 24, 36, 48], [6, 18, 30, 42]],
            N_tp=1763,
            ell_tp=13,
            B_mat=[[3, 1, 0, 1], [1, 2, 1, 0], [0, 1, 3, 1], [1, 0, 1, 2]],
            Q_lcm_hint=2,
        )
    )
    # 12. m=4 ModHNF (M=96 precision) + 5x5 bordered adjugate + TryPair mod N=3599 (59*61, ell=17)
    certs.append(
        make_mode3_cert(
            "m3_fp_hnf4_trypair5_n3599",
            "High-precision Lemma 2.2 (M=96) + 4D ModHNF + 5x5 TryPair annihilation mod N=3599",
            al_params=(65537, 257, 96),
            asq_params=(10007, 101, 96),
            hnf_m=4,
            hnf_dmod=210,
            hnf_cols=[[15, 30, 45, 60], [14, 21, 35, 49], [10, 20, 40, 50], [6, 12, 18, 24], [30, 60, 90, 120]],
            N_tp=3599,
            ell_tp=17,
            B_mat=[
                [2, 1, 0, 0, 1],
                [1, 3, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [1, 0, 0, 1, 2],
            ],
            Q_lcm_hint=5,
        )
    )

    # ---- MODE 4: 4 Canonical Certificates (Lemma 8.1 Base-m Pair + Lemma 11.1 Saturation + Factoring) ----
    # Need N = p1 * p2 with m = ceil(N^(1/d)) >= 4*d and gcd(N, P0 * ell) = 1 and min(p1, p2) > K_idx!
    # For d = 4, z0 = 3 (P0=6), m >= 16 => N > 15^4 = 50,625!
    # 13. p1 = 251, p2 = 257 => N = 64,507 (m = 16, 15^4=50625 < 64507 < 65536=16^4!), ell=7, K_idx=20
    certs.append(
        make_mode4_cert(
            "m4_basem_sat_n64507_d4",
            "N=251*257=64507 (d=4, m=16): Lemma 8.1 base-m pair + Lemma 11.1 saturation (K_idx=20) + factoring",
            p1=251,
            p2=257,
            d=4,
            z0=3,
            ell=7,
            P_ok=[11, 13, 17, 19],
            K_idx=20,
        )
    )
    # 14. p1 = 503, p2 = 509 => N = 256,027 (m = 23, 22^4=234256 < 256027 < 279841=23^4), ell=11, K_idx=30
    certs.append(
        make_mode4_cert(
            "m4_basem_sat_n256027_d4",
            "N=503*509=256027 (d=4, m=23): Lemma 8.1 base-m pair + Lemma 11.1 saturation (K_idx=30) + factoring",
            p1=503,
            p2=509,
            d=4,
            z0=3,
            ell=11,
            P_ok=[13, 17, 19, 23, 29],
            K_idx=30,
        )
    )
    # 15. p1 = 1009, p2 = 1013 => N = 1,022,117 (m = 32, 31^4=923521 < 1022117 < 1048576=32^4), ell=13, K_idx=40
    certs.append(
        make_mode4_cert(
            "m4_basem_sat_n1022117_d4",
            "N=1009*1013=1022117 (d=4, m=32): Lemma 8.1 base-m pair + Lemma 11.1 saturation (K_idx=40) + factoring",
            p1=1009,
            p2=1013,
            d=4,
            z0=3,
            ell=13,
            P_ok=[17, 19, 23, 29, 31],
            K_idx=40,
        )
    )
    # 16. d = 5! Need m >= 20 => N > 19^5 = 2,476,099! Take p1 = 2003, p2 = 2011 => N = 4,028,033 (m=21, 20^5=3200000 < N < 4084101=21^5), ell=7, K_idx=50
    certs.append(
        make_mode4_cert(
            "m4_basem_sat_n4028033_d5",
            "N=2003*2011=4028033 (d=5, m=21): Lemma 8.1 degree-5 base-m pair + Lemma 11.1 saturation (K_idx=50)",
            p1=2003,
            p2=2011,
            d=5,
            z0=3,
            ell=7,
            P_ok=[11, 13, 17, 19, 23, 29],
            K_idx=50,
        )
    )

    return certs


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    for old in CERT_DIR.glob("*.cert.json"):
        old.unlink()

    certs = build_all_16_certificates()
    assert len(certs) == 16

    # Exp 1: Write, verify, and benchmark all 16 certificates (7-run mean after warm-up)
    atlas_rows: List[Dict[str, Any]] = []
    for cert in certs:
        out_path = CERT_DIR / f"{cert['cert_id']}.cert.json"
        out_path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")

        res = verify_certificate_dict(cert)
        assert res["status"] == "VERIFIED"

        times_ms: List[float] = []
        for _ in range(7):
            t0 = time.perf_counter()
            verify_certificate_dict(cert)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = round(sum(times_ms) / len(times_ms), 3)
        atlas_rows.append(
            {
                "cert_id": cert["cert_id"],
                "mode": cert["mode"],
                "mean_ms": mean_ms,
                "min_ms": round(min(times_ms), 3),
                "max_ms": round(max(times_ms), 3),
                "summary": res["summary"],
            }
        )
        print(f"[VERIFIED] {cert['cert_id']:36s} ({cert['mode']:34s}) {mean_ms:7.3f} ms")

    (RES_DIR / "exp1_certificate_atlas.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "count": len(atlas_rows), "certificates": atlas_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # Exp 2: Asymptotic Complexity Exponent Comparison (log2 bit-ops vs log2 N)
    # Compare: Pollard-Strassen N^{1/4}, Harvey (2021) N^{1/5}, Burgess-Vinogradov QNR p^{1/(4 sqrt(e))},
    # Cogentic Factoring exp(3 (ln N)^{3/4}), and Cogentic QNR exp((ln p)^{1/2} ln ln p) (unit-constant leading exponents).
    ln2 = log(2.0)
    bv_exp = 1.0 / (4.0 * exp(0.5))  # 0.15163266...
    comp_rows: List[Dict[str, Any]] = []
    for bits in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        L = bits * ln2
        log2_harvey = 0.20 * bits
        log2_ps = 0.25 * bits
        log2_bv_qnr = bv_exp * bits
        log2_cog_fac = (3.0 * (L ** 0.75)) / ln2
        log2_cog_qnr = ((L ** 0.5) * log(L + 16.0)) / ln2
        comp_rows.append(
            {
                "bits_log2_N": bits,
                "log2_pollard_strassen_N1_4": round(log2_ps, 2),
                "log2_harvey_N1_5": round(log2_harvey, 2),
                "log2_burgess_vinogradov_QNR": round(log2_bv_qnr, 2),
                "log2_cogentic_exp_3_L34": round(log2_cog_fac, 2),
                "log2_cogentic_L12_QNR_unitC": round(log2_cog_qnr, 2),
                "cogentic_fac_beats_harvey": log2_cog_fac < log2_harvey,
            }
        )
    # Exact crossover where 3 (ln N)^{3/4} = 0.20 ln N => (ln N)^{1/4} = 15 => ln N = 50625 => log2 N ~ 73036.4
    crossover_ln_N = 15.0 ** 4
    crossover_bits = round(crossover_ln_N / ln2, 2)
    (RES_DIR / "exp2_complexity_crossover.json").write_text(
        json.dumps(
            {
                "burgess_vinogradov_exponent": round(bv_exp, 6),
                "harvey_vs_cogentic_lnN_crossover": crossover_ln_N,
                "harvey_vs_cogentic_bits_crossover": crossover_bits,
                "sweep": comp_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Exp 3: Exact Walsh-Hadamard Amplification Sweep across q in {71, 311, 479, 1559} and K in {1, 2, 3, 4}
    wh_rows: List[Dict[str, Any]] = []
    for q_p, Y_p, X_p in [(71, 6, 120), (311, 10, 250), (479, 12, 400), (1559, 16, 600)]:
        for K_val in [1, 2, 3, 4]:
            rm = miller_multiquadratic_roots(q_p, q_p - 1, K_val)
            if rm["outcome"] != "COMPATIBLE_FAMILY":
                continue
            wh = walsh_hadamard_norm_sums_exact(q_p, rm["family"], K_val, X_p, Y_p)
            wh_rows.append(
                {
                    "q_prime": q_p,
                    "K": K_val,
                    "num_fields": (1 << K_val) - 1,
                    "X": X_p,
                    "Y": Y_p,
                    "smooth_Psi": wh["smooth_count_Psi"],
                    "b0_total": wh["b0_total"],
                    "max_S_ell": wh["max_S_ell"],
                    "total_S_ell": wh["total_S_ell"],
                    "wh_rhs_lower": (1 << K_val) * wh["smooth_count_Psi"] - wh["b0_total"],
                    "wh_lhs_upper": ((1 << K_val) - 1) * wh["max_S_ell"],
                }
            )
    (RES_DIR / "exp3_walsh_hadamard_amplification.json").write_text(
        json.dumps({"sweep": wh_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Exp 4: Real-Quadratic Gauss Reduction Cycles & Regulator Growth across squarefree d <= 60
    gauss_rows: List[Dict[str, Any]] = []
    for d_val in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 29, 31, 37, 41, 43, 46, 53, 58, 61]:
        gr = gauss_reduction_cycle_real(d_val)
        gauss_rows.append(
            {
                "d": d_val,
                "D": gr["D"],
                "period_L": gr["period_L"],
                "unit_norm": gr["unit_norm"],
                "unit_x_bits": abs(gr["unit_x"]).bit_length(),
                "unit_y_bits": abs(gr["unit_y"]).bit_length(),
            }
        )
    (RES_DIR / "exp4_gauss_cycles.json").write_text(
        json.dumps({"cycles": gauss_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nGenerated 16 certificates and 4 result JSONs successfully.")


if __name__ == "__main__":
    main()
