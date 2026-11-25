"""Zero-float 4-mode certificate verifier for Paper 61 (DETFACT-COGENTIC-CERT-1.0).

Verifies exact Miller 2-Sylow multiquadratic square-root compatibility, quadratic
infrastructure Gauss cycles & Walsh-Hadamard positivity + QNR witnesses, exact
fixed-point ApproxLog/ApproxSqrt + ModHNF + TryPair two-field annihilation, and
preconditioned base-m polynomial pairs + subgroup-index saturation + composite
factoring for Cogentic (2026) O(exp(3 (ln N)^(3/4))) & L_p[1/2, O(1)].

Strictly zero floating-point arithmetic (`float`, `math.log`, `numpy` forbidden).
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.detfact_exact import (
    approx_log_exact,
    approx_sqrt_exact,
    check_biquadratic_compatibility,
    exact_det_and_adjugate,
    first_k_primes,
    fmt_frac,
    fundamental_discriminant,
    gauss_reduction_cycle_real,
    is_irreducible_f2,
    is_prime_exact,
    is_squarefree,
    lcm_1_to_k,
    legendre_symbol,
    miller_multiquadratic_roots,
    mod_hnf_exact,
    parse_exact_fraction,
    parse_exact_int,
    poly_deriv_int,
    poly_eval_int,
    poly_eval_mod,
    walsh_hadamard_norm_sums_exact,
)


SCHEMA_VERSION = "DETFACT-COGENTIC-CERT-1.0"
MAX_JSON_DEPTH = 16
MAX_JSON_BYTES = 2_000_000

VALID_MODES = {
    "miller_multiquadratic_compat_cert",
    "quadratic_infrastructure_wh_cert",
    "fixedpoint_hnf_and_trypair_cert",
    "base_m_saturation_and_factor_cert",
}


class CertificateError(ValueError):
    """Raised when a certificate fails structural, parsing, or mathematical verification."""


# ---------------------------------------------------------------------------
# Strict JSON Loader (Duplicate-Key Rejection, Zero-Float, Depth Limit)
# ---------------------------------------------------------------------------


def _no_duplicate_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for k, v in pairs:
        if k in obj:
            raise CertificateError(f"Duplicate JSON key detected: {k!r}")
        obj[k] = v
    return obj


def _reject_float_constant(val: str) -> Any:
    raise CertificateError(f"Floating-point constant rejected in strict zero-float certificate: {val!r}")


def _check_depth_and_types(obj: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise CertificateError(f"JSON nesting depth exceeded maximum {MAX_JSON_DEPTH}")
    if isinstance(obj, float):
        raise CertificateError(f"Floating-point value {obj!r} rejected in strict zero-float certificate")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise CertificateError(f"Non-string JSON object key: {k!r}")
            _check_depth_and_types(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth_and_types(item, depth + 1)


def load_strict_json(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise CertificateError("Certificate input must be a UTF-8 string within size limit")
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_pairs,
            parse_float=_reject_float_constant,
            parse_constant=_reject_float_constant,
        )
    except CertificateError:
        raise
    except Exception as exc:
        raise CertificateError(f"Invalid JSON syntax: {exc}") from exc
    if not isinstance(data, dict):
        raise CertificateError("Top-level certificate must be a JSON object")
    _check_depth_and_types(data)
    return data


def _require_keys(obj: Dict[str, Any], required: Set[str], optional: Set[str], ctx: str) -> None:
    keys = set(obj.keys())
    missing = required - keys
    extra = keys - (required | optional)
    if missing:
        raise CertificateError(f"{ctx}: missing required keys {sorted(missing)}")
    if extra:
        raise CertificateError(f"{ctx}: unexpected extra keys {sorted(extra)}")


def _require_bool(val: Any, field: str) -> bool:
    if not isinstance(val, bool):
        raise CertificateError(f"{field}: expected boolean, got {type(val).__name__}")
    return val


def _require_str(val: Any, field: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise CertificateError(f"{field}: expected non-empty string")
    return val


# ---------------------------------------------------------------------------
# Mode 1: miller_multiquadratic_compat_cert (Sec. 3: Miller 2-Sylow + Biquadratic Roots)
# ---------------------------------------------------------------------------


def _verify_mode1(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={"N", "M_tilde", "K", "claimed_outcome"},
        optional={
            "claimed_divisor",
            "claimed_j_star",
            "claimed_t_star",
            "claimed_family",
            "note",
        },
        ctx="mode1.payload",
    )

    N = parse_exact_int(payload["N"], "N")
    M_tilde = parse_exact_int(payload["M_tilde"], "M_tilde")
    K = parse_exact_int(payload["K"], "K")
    if N < 9 or N % 2 == 0 or N.bit_length() > 256:
        raise CertificateError(f"Mode 1 requires odd N >= 9 within 256-bit cap, got N={N}")
    if M_tilde < 2 or M_tilde % 2 != 0 or M_tilde.bit_length() > 2048:
        raise CertificateError(f"Mode 1 requires even M_tilde >= 2 within 2048-bit cap, got {M_tilde}")
    if not (2 <= K <= 5):
        raise CertificateError(f"Mode 1 requires 2 <= K <= 5, got K={K}")

    claimed_outcome = _require_str(payload["claimed_outcome"], "claimed_outcome")
    if claimed_outcome not in {"SPLIT", "COMPATIBLE_FAMILY"}:
        raise CertificateError(f"Invalid Mode 1 claimed_outcome: {claimed_outcome!r}")

    res = miller_multiquadratic_roots(N, M_tilde, K)
    if res["outcome"] != claimed_outcome:
        raise CertificateError(
            f"claimed_outcome mismatch: claimed {claimed_outcome!r}, actual {res['outcome']!r}"
        )

    if claimed_outcome == "SPLIT":
        if "claimed_divisor" not in payload:
            raise CertificateError("SPLIT outcome requires claimed_divisor")
        d = parse_exact_int(payload["claimed_divisor"], "claimed_divisor")
        if not (1 < d < N) or N % d != 0:
            raise CertificateError(f"claimed_divisor {d} is not a nontrivial divisor of N={N}")
        return {
            "N": N,
            "K": K,
            "outcome": "SPLIT",
            "divisor": d,
            "cofactor": N // d,
        }

    # COMPATIBLE_FAMILY: verify full 2^K - 1 family + all pairwise biquadratic compatibility relations
    for req_k in ("claimed_j_star", "claimed_t_star", "claimed_family"):
        if req_k not in payload:
            raise CertificateError(f"COMPATIBLE_FAMILY outcome requires {req_k}")

    cl_j = parse_exact_int(payload["claimed_j_star"], "claimed_j_star")
    cl_t = parse_exact_int(payload["claimed_t_star"], "claimed_t_star")
    if cl_j != res["j_star"]:
        raise CertificateError(f"claimed_j_star mismatch: claimed {cl_j}, actual {res['j_star']}")
    if cl_t != res["t_star"]:
        raise CertificateError(f"claimed_t_star mismatch: claimed {cl_t}, actual {res['t_star']}")

    cl_fam_raw = payload["claimed_family"]
    if not isinstance(cl_fam_raw, dict):
        raise CertificateError("claimed_family must be a JSON object")
    expected_keys = {format(m, f"0{K}b")[::-1] for m in range(1, 1 << K)}
    if set(cl_fam_raw.keys()) != expected_keys:
        raise CertificateError(f"claimed_family keys must be exact 2^K - 1 bitstrings of length {K}")

    cl_fam: Dict[str, Dict[str, int]] = {}
    for k, entry in cl_fam_raw.items():
        if not isinstance(entry, dict) or set(entry.keys()) != {"d", "s"}:
            raise CertificateError(f"claimed_family[{k!r}] must have exact keys {{'d', 's'}}")
        d_val = parse_exact_int(entry["d"], f"claimed_family[{k}].d")
        s_val = parse_exact_int(entry["s"], f"claimed_family[{k}].s")
        cl_fam[k] = {"d": d_val, "s": s_val}

    if not check_biquadratic_compatibility(N, cl_fam, K):
        raise CertificateError("claimed_family failed pairwise biquadratic compatibility s_l1*s_l2 == eta12*s_l3 (mod N)")
    if cl_fam != res["family"]:
        raise CertificateError("claimed_family does not match canonical Miller 2-Sylow multiquadratic extraction")

    num_pairs = ((1 << K) - 1) * ((1 << K) - 2) // 2
    return {
        "N": N,
        "K": K,
        "outcome": "COMPATIBLE_FAMILY",
        "j_star": res["j_star"],
        "t_star": res["t_star"],
        "family_size": len(cl_fam),
        "biquadratic_pairs_verified": num_pairs,
    }


# ---------------------------------------------------------------------------
# Mode 2: quadratic_infrastructure_wh_cert (Sec. 4–5: Gauss Cycles, Walsh-Hadamard & QNR)
# ---------------------------------------------------------------------------


def _verify_mode2(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "q_prime",
            "K",
            "X",
            "Y",
            "family",
            "gauss_real_d",
            "claimed_gauss_period",
            "claimed_unit_x",
            "claimed_unit_y",
            "claimed_unit_norm",
            "claimed_smooth_Psi",
            "claimed_max_S_ell",
            "claimed_qnr_z",
        },
        optional={"note"},
        ctx="mode2.payload",
    )

    q_prime = parse_exact_int(payload["q_prime"], "q_prime")
    K = parse_exact_int(payload["K"], "K")
    X = parse_exact_int(payload["X"], "X")
    Y = parse_exact_int(payload["Y"], "Y")
    if q_prime < 11 or not is_prime_exact(q_prime):
        raise CertificateError(f"Mode 2 requires odd prime q_prime >= 11, got {q_prime}")
    if not (2 <= K <= 4):
        raise CertificateError(f"Mode 2 requires 2 <= K <= 4, got K={K}")
    if not (2 <= Y <= X <= 5000):
        raise CertificateError(f"Mode 2 requires 2 <= Y <= X <= 5000, got Y={Y}, X={X}")

    # Verify exact QNR witness z in {1, ..., q_prime - 1} with (z / q_prime) = -1 (Theorem 1.2(b))
    z = parse_exact_int(payload["claimed_qnr_z"], "claimed_qnr_z")
    if not (1 <= z <= q_prime - 1):
        raise CertificateError(f"claimed_qnr_z={z} out of range [1, {q_prime - 1}]")
    if legendre_symbol(z, q_prime) != -1:
        raise CertificateError(f"claimed_qnr_z={z} is NOT a quadratic nonresidue modulo {q_prime}")

    # Verify multiquadratic family compatibility & exact {-1, q_1..q_K} support modulo q_prime
    fam_raw = payload["family"]
    if not isinstance(fam_raw, dict):
        raise CertificateError("family must be a JSON object")
    expected_keys = {format(m, f"0{K}b")[::-1] for m in range(1, 1 << K)}
    if set(fam_raw.keys()) != expected_keys:
        raise CertificateError(f"family keys must be exact 2^K - 1 bitstrings of length {K}")
    primes_K = first_k_primes(K)
    PK = 1
    for p_k in primes_K:
        PK *= p_k
    fam: Dict[str, Dict[str, int]] = {}
    for k, entry in fam_raw.items():
        if not isinstance(entry, dict) or set(entry.keys()) != {"d", "s"}:
            raise CertificateError(f"family[{k!r}] must have exact keys {{'d', 's'}}")
        d_val = parse_exact_int(entry["d"], f"family[{k}].d")
        s_val = parse_exact_int(entry["s"], f"family[{k}].s")
        if d_val == 0 or PK % abs(d_val) != 0:
            raise CertificateError(f"family[{k}].d={d_val} must be supported on {{-1}} U first K primes (divide P_K={PK})")
        fam[k] = {"d": d_val, "s": s_val}
    if not check_biquadratic_compatibility(q_prime, fam, K):
        raise CertificateError("family failed biquadratic compatibility modulo q_prime")

    # Verify exact real-quadratic Gauss reduction cycle & fundamental unit (Lemma 4.2(1))
    gd = parse_exact_int(payload["gauss_real_d"], "gauss_real_d")
    if gd <= 1 or gd > 10_000 or not is_squarefree(gd):
        raise CertificateError(f"gauss_real_d={gd} must be squarefree in (1, 10000]")
    gres = gauss_reduction_cycle_real(gd)
    if parse_exact_int(payload["claimed_gauss_period"], "claimed_gauss_period") != gres["period_L"]:
        raise CertificateError(f"claimed_gauss_period mismatch: actual {gres['period_L']}")
    if parse_exact_int(payload["claimed_unit_x"], "claimed_unit_x") != gres["unit_x"]:
        raise CertificateError(f"claimed_unit_x mismatch: actual {gres['unit_x']}")
    if parse_exact_int(payload["claimed_unit_y"], "claimed_unit_y") != gres["unit_y"]:
        raise CertificateError(f"claimed_unit_y mismatch: actual {gres['unit_y']}")
    if parse_exact_int(payload["claimed_unit_norm"], "claimed_unit_norm") != gres["unit_norm"]:
        raise CertificateError(f"claimed_unit_norm mismatch: actual {gres['unit_norm']}")

    # Verify exact Walsh-Hadamard positivity & Lemma 5.1(3) amplification inequality
    wh = walsh_hadamard_norm_sums_exact(q_prime, fam, K, X, Y)
    if not wh["chi_on_Y_all_plus"]:
        raise CertificateError(f"Legendre character chi=(./{q_prime}) is not identically +1 on [1, Y={Y}]")
    if not wh["all_wh_nonneg"]:
        raise CertificateError("Walsh-Hadamard nonnegativity violated")
    if not wh["all_smooth_ge_1"]:
        raise CertificateError("Y-smooth amplification b_hat_0(n) >= 1 violated")
    if not wh["lemma51_3_holds"]:
        raise CertificateError("Lemma 5.1(3) Walsh-Hadamard sum inequality violated")

    if parse_exact_int(payload["claimed_smooth_Psi"], "claimed_smooth_Psi") != wh["smooth_count_Psi"]:
        raise CertificateError(f"claimed_smooth_Psi mismatch: actual {wh['smooth_count_Psi']}")
    if parse_exact_int(payload["claimed_max_S_ell"], "claimed_max_S_ell") != wh["max_S_ell"]:
        raise CertificateError(f"claimed_max_S_ell mismatch: actual {wh['max_S_ell']}")

    return {
        "q_prime": q_prime,
        "K": K,
        "X": X,
        "Y": Y,
        "qnr_z": z,
        "gauss_real_d": gd,
        "gauss_disc_D": gres["D"],
        "gauss_period_L": gres["period_L"],
        "unit_norm": gres["unit_norm"],
        "smooth_Psi": wh["smooth_count_Psi"],
        "b0_total": wh["b0_total"],
        "max_S_ell": wh["max_S_ell"],
    }


# ---------------------------------------------------------------------------
# Mode 3: fixedpoint_hnf_and_trypair_cert (Sec. 2.3 & 7.3–7.4: ApproxLog/Sqrt, ModHNF, TryPair)
# ---------------------------------------------------------------------------


def _verify_mode3(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "approx_log_probe",
            "approx_sqrt_probe",
            "mod_hnf_probe",
            "trypair_annihilation_probe",
        },
        optional={"note"},
        ctx="mode3.payload",
    )

    # 1. Lemma 2.2(1) ApproxLog exact verification
    al = payload["approx_log_probe"]
    if not isinstance(al, dict):
        raise CertificateError("approx_log_probe must be an object")
    _require_keys(al, required={"u", "v", "M", "claimed_num"}, optional=set(), ctx="approx_log_probe")
    u_l = parse_exact_int(al["u"], "approx_log_probe.u")
    v_l = parse_exact_int(al["v"], "approx_log_probe.v")
    M_l = parse_exact_int(al["M"], "approx_log_probe.M")
    if M_l < 4 or M_l > 256:
        raise CertificateError(f"approx_log_probe.M must lie in [4, 256], got {M_l}")
    num_l, _ = approx_log_exact(u_l, v_l, M_l)
    if parse_exact_int(al["claimed_num"], "approx_log_probe.claimed_num") != num_l:
        raise CertificateError(f"approx_log_probe.claimed_num mismatch: actual {num_l}")

    # 2. Lemma 2.2(3) ApproxSqrt exact verification + direct exact-integer bracketing proof!
    asq = payload["approx_sqrt_probe"]
    if not isinstance(asq, dict):
        raise CertificateError("approx_sqrt_probe must be an object")
    _require_keys(asq, required={"u", "v", "M", "claimed_num"}, optional=set(), ctx="approx_sqrt_probe")
    u_s = parse_exact_int(asq["u"], "approx_sqrt_probe.u")
    v_s = parse_exact_int(asq["v"], "approx_sqrt_probe.v")
    M_s = parse_exact_int(asq["M"], "approx_sqrt_probe.M")
    if M_s < 2 or M_s > 256:
        raise CertificateError(f"approx_sqrt_probe.M must lie in [2, 256], got {M_s}")
    num_s, _ = approx_sqrt_exact(u_s, v_s, M_s)
    if parse_exact_int(asq["claimed_num"], "approx_sqrt_probe.claimed_num") != num_s:
        raise CertificateError(f"approx_sqrt_probe.claimed_num mismatch: actual {num_s}")
    # Direct exact zero-float verification of |sqrt(u/v) - num_s * 2^(-M_s)| < 2^(-M_s):
    # (num_s - 1)^2 * v < u * 2^(2*M_s) < (num_s + 1)^2 * v  (for num_s >= 1)
    lhs_low = (max(0, num_s - 1) ** 2) * v_s
    mid = u_s << (2 * M_s)
    rhs_hi = ((num_s + 1) ** 2) * v_s
    if not (lhs_low <= mid < rhs_hi):
        raise CertificateError("ApproxSqrt exact bracketing inequality failed")

    # 3. Lemma 2.2(4) ModHNF exact verification in Z^m (1 <= m <= 4)
    hnf = payload["mod_hnf_probe"]
    if not isinstance(hnf, dict):
        raise CertificateError("mod_hnf_probe must be an object")
    _require_keys(hnf, required={"m", "d_mod", "columns", "claimed_H"}, optional=set(), ctx="mod_hnf_probe")
    m_dim = parse_exact_int(hnf["m"], "mod_hnf_probe.m")
    d_mod = parse_exact_int(hnf["d_mod"], "mod_hnf_probe.d_mod")
    if not (1 <= m_dim <= 4) or d_mod < 1:
        raise CertificateError(f"ModHNF requires 1 <= m <= 4 and d_mod >= 1, got m={m_dim}, d_mod={d_mod}")
    cols_raw = hnf["columns"]
    cl_H_raw = hnf["claimed_H"]
    if not isinstance(cols_raw, list) or not cols_raw:
        raise CertificateError("mod_hnf_probe.columns must be a non-empty list")
    cols: List[List[int]] = []
    for i, col in enumerate(cols_raw):
        if not isinstance(col, list) or len(col) != m_dim:
            raise CertificateError(f"mod_hnf_probe.columns[{i}] must be a list of length {m_dim}")
        cols.append([parse_exact_int(x, f"columns[{i}][{j}]") for j, x in enumerate(col)])
    H_actual = mod_hnf_exact(cols, m_dim, d_mod)
    if not isinstance(cl_H_raw, list) or len(cl_H_raw) != m_dim:
        raise CertificateError("mod_hnf_probe.claimed_H must be a list of m columns")
    H_claimed: List[List[int]] = []
    for i, col in enumerate(cl_H_raw):
        if not isinstance(col, list) or len(col) != m_dim:
            raise CertificateError(f"mod_hnf_probe.claimed_H[{i}] must be a list of length {m_dim}")
        H_claimed.append([parse_exact_int(x, f"claimed_H[{i}][{j}]") for j, x in enumerate(col)])
    if H_claimed != H_actual:
        raise CertificateError(f"mod_hnf_probe.claimed_H mismatch: actual {H_actual}")
    # Verify upper-triangular HNF invariants explicitly
    hnf_det = 1
    for i in range(m_dim):
        if H_actual[i][i] < 1 or d_mod % H_actual[i][i] != 0:
            raise CertificateError("HNF diagonal divisibility invariant failed")
        hnf_det *= H_actual[i][i]
        for r in range(i + 1, m_dim):
            if H_actual[i][r] != 0:
                raise CertificateError("HNF not upper-triangular")
        for j in range(i + 1, m_dim):
            if not (0 <= H_actual[j][i] < H_actual[i][i]):
                raise CertificateError("HNF off-diagonal reduction invariant failed")

    # 4. Lemmas 7.4–7.5 Bordered-Adjugate + Two-Field Localization Annihilation Probe
    tp = payload["trypair_annihilation_probe"]
    if not isinstance(tp, dict):
        raise CertificateError("trypair_annihilation_probe must be an object")
    _require_keys(
        tp,
        required={
            "N",
            "ell",
            "pivot_matrix_B",
            "claimed_det_B",
            "claimed_adj_B",
            "E0",
            "Q_lcm",
            "P_mod_N",
            "s1",
            "s2",
            "claimed_E_ell",
        },
        optional=set(),
        ctx="trypair_annihilation_probe",
    )
    N_tp = parse_exact_int(tp["N"], "trypair.N")
    ell = parse_exact_int(tp["ell"], "trypair.ell")
    if N_tp < 9 or N_tp % 2 == 0 or ell < 2 or not is_prime_exact(ell) or gcd(ell, N_tp) != 1:
        raise CertificateError("trypair requires odd N >= 9 and prime ell coprime to N")

    B_raw = tp["pivot_matrix_B"]
    if not isinstance(B_raw, list) or not (2 <= len(B_raw) <= 6):
        raise CertificateError("trypair.pivot_matrix_B must be k x k with 2 <= k <= 6")
    k_sz = len(B_raw)
    B_mat = []
    for i, row in enumerate(B_raw):
        if not isinstance(row, list) or len(row) != k_sz:
            raise CertificateError("trypair.pivot_matrix_B must be square")
        B_mat.append([parse_exact_int(x, f"B[{i}][{j}]") for j, x in enumerate(row)])

    det_B, adj_B = exact_det_and_adjugate(B_mat)
    if det_B == 0:
        raise CertificateError("trypair.pivot_matrix_B must be nonsingular (det != 0)")
    cl_det = parse_exact_int(tp["claimed_det_B"], "trypair.claimed_det_B")
    if cl_det != det_B:
        raise CertificateError(f"trypair.claimed_det_B mismatch: claimed {cl_det}, actual {det_B}")

    cl_adj_raw = tp["claimed_adj_B"]
    if not isinstance(cl_adj_raw, list) or len(cl_adj_raw) != k_sz:
        raise CertificateError("trypair.claimed_adj_B must be square matching pivot_matrix_B")
    cl_adj: List[List[int]] = []
    for i, row in enumerate(cl_adj_raw):
        if not isinstance(row, list) or len(row) != k_sz:
            raise CertificateError(f"trypair.claimed_adj_B[{i}] must be a list of length {k_sz}")
        cl_adj.append([parse_exact_int(x, f"adj[{i}][{j}]") for j, x in enumerate(row)])
    if cl_adj != adj_B:
        raise CertificateError("trypair.claimed_adj_B mismatch")
    # Independently check B * adj(B) == det(B) * I_k
    for i in range(k_sz):
        for j in range(k_sz):
            dot = sum(B_mat[i][t] * adj_B[t][j] for t in range(k_sz))
            expected = det_B if i == j else 0
            if dot != expected:
                raise CertificateError("Bordered-adjugate matrix identity B * adj(B) = det(B) * I failed")

    E0 = parse_exact_int(tp["E0"], "trypair.E0")
    Q_lcm = parse_exact_int(tp["Q_lcm"], "trypair.Q_lcm")
    if E0 != abs(det_B):
        raise CertificateError(f"trypair.E0 must equal |det(B)|={abs(det_B)}, got {E0}")
    if Q_lcm < 1:
        raise CertificateError("trypair.Q_lcm must be >= 1")
    P_mod = parse_exact_int(tp["P_mod_N"], "trypair.P_mod_N") % N_tp
    if gcd(P_mod, N_tp) != 1:
        raise CertificateError("trypair.P_mod_N must be a unit modulo N")
    s1 = parse_exact_int(tp["s1"], "trypair.s1")
    s2 = parse_exact_int(tp["s2"], "trypair.s2")
    if s1 not in (-1, 1) or s2 not in (-1, 1):
        raise CertificateError("trypair signs s1, s2 must lie in {-1, +1}")

    E_half = E0 * Q_lcm
    # Check exact Lemma 7.4(2) localization congruences:
    # ell^{E_half} * P == s1 (mod N) and P == s2 (mod N) => ell^{2 * E_half} == 1 (mod N)
    if (pow(ell, E_half, N_tp) * P_mod - s1) % N_tp != 0:
        raise CertificateError("Lemma 7.4(2) field-1 localization congruence ell^E * P == s1 (mod N) failed")
    if (P_mod - s2) % N_tp != 0:
        raise CertificateError("Lemma 7.4(2) field-2 localization congruence P == s2 (mod N) failed")

    E_ell = 2 * E_half
    cl_E_ell = parse_exact_int(tp["claimed_E_ell"], "trypair.claimed_E_ell")
    if cl_E_ell != E_ell:
        raise CertificateError(f"trypair.claimed_E_ell mismatch: claimed {cl_E_ell}, actual {E_ell}")
    if pow(ell, E_ell, N_tp) != 1:
        raise CertificateError("TryPair final annihilation check ell^{E_ell} == 1 (mod N) failed")

    return {
        "approx_log_num": num_l,
        "approx_log_M": M_l,
        "approx_sqrt_num": num_s,
        "approx_sqrt_M": M_s,
        "hnf_dim": m_dim,
        "hnf_det": hnf_det,
        "trypair_det_B": det_B,
        "trypair_E_ell": E_ell,
    }


# ---------------------------------------------------------------------------
# Mode 4: base_m_saturation_and_factor_cert (Sec. 8, 11, Thm 1.1 & 1.3)
# ---------------------------------------------------------------------------


def _verify_mode4(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "N",
            "factor_p1",
            "factor_p2",
            "base_m_pair",
            "saturation_probe",
        },
        optional={"note"},
        ctx="mode4.payload",
    )

    N = parse_exact_int(payload["N"], "N")
    p1 = parse_exact_int(payload["factor_p1"], "factor_p1")
    p2 = parse_exact_int(payload["factor_p2"], "factor_p2")
    if N < 9 or N % 2 == 0:
        raise CertificateError(f"Mode 4 requires odd composite N >= 9, got N={N}")
    if not (1 < p1 < N and 1 < p2 < N and p1 * p2 == N):
        raise CertificateError(f"Invalid nontrivial factorization: {p1} * {p2} != {N}")
    if not (is_prime_exact(p1) and is_prime_exact(p2)) or p1 == p2:
        raise CertificateError("Mode 4 requires non-perfect-power semiprime N = p1 * p2 (p1 != p2 primes)")

    # 1. Lemma 8.1 Preconditioned Base-m Polynomial Pair Audit
    bm = payload["base_m_pair"]
    if not isinstance(bm, dict):
        raise CertificateError("base_m_pair must be an object")
    _require_keys(
        bm,
        required={"d", "z0", "m", "ell", "f1_coeffs", "f2_coeffs", "claimed_k0", "claimed_k0_prime"},
        optional=set(),
        ctx="base_m_pair",
    )
    d = parse_exact_int(bm["d"], "base_m_pair.d")
    z0 = parse_exact_int(bm["z0"], "base_m_pair.z0")
    m = parse_exact_int(bm["m"], "base_m_pair.m")
    ell = parse_exact_int(bm["ell"], "base_m_pair.ell")
    if not (4 <= d <= 16) or not (3 <= z0 <= 11) or m < 4 * d:
        raise CertificateError("Lemma 8.1 requires 4 <= d <= 16, 3 <= z0 <= 11, m >= 4*d")
    if ell <= max(z0, d) or ell > 10_000 or not is_prime_exact(ell):
        raise CertificateError("Lemma 8.1 requires prime ell in (max(z0, d), 10000]")
    if not ((m - 1) ** d < N < m ** d):
        raise CertificateError(f"Lemma 8.1 base-m bracket (m-1)^d < N < m^d failed for m={m}, d={d}")

    # Compute P0 = prod_{p <= z0} p and Podd = P0 // 2
    P0 = 1
    for p in range(2, z0 + 1):
        if is_prime_exact(p):
            P0 *= p
    Podd = P0 // 2
    if gcd(N, P0 * ell) != 1:
        raise CertificateError("Lemma 8.1 requires gcd(N, P0 * ell) = 1")

    f1_raw = bm["f1_coeffs"]
    f2_raw = bm["f2_coeffs"]
    if not isinstance(f1_raw, list) or not isinstance(f2_raw, list) or len(f1_raw) != d + 1 or len(f2_raw) != d + 1:
        raise CertificateError(f"f1_coeffs and f2_coeffs must have length d+1={d+1}")
    f1 = [parse_exact_int(c, f"f1[{j}]") for j, c in enumerate(f1_raw)]
    f2 = [parse_exact_int(c, f"f2[{j}]") for j, c in enumerate(f2_raw)]

    cd = f1[-1]
    if f2[-1] != cd or cd < 4 * P0 * ell or cd % Podd != 0 or cd % 2 == 0:
        raise CertificateError("Lemma 8.1 leading coefficient cd condition failed (shared odd cd >= 4*P0*ell, Podd | cd)")

    # Check exact irreducibility mod 2 (Lemma 8.1(3): f_i mod 2 monic irreducible of degree d in F_2[X])
    if not is_irreducible_f2(f1):
        raise CertificateError("Lemma 8.1(3) failed: f1 mod 2 is not irreducible of degree d in F_2[X]")
    if not is_irreducible_f2(f2):
        raise CertificateError("Lemma 8.1(3) failed: f2 mod 2 is not irreducible of degree d in F_2[X]")

    # Check f_i(m) == 0 (mod N) and k0, k0' coprime to P0 * ell
    v1_m = poly_eval_int(f1, m)
    v2_m = poly_eval_int(f2, m)
    if v1_m % N != 0 or v2_m % N != 0:
        raise CertificateError("Lemma 8.1(2) failed: f_i(m) != 0 (mod N)")
    k0 = v1_m // N
    k0_p = v2_m // N
    if not (1 <= k0 < 2 * cd and 1 <= k0_p < 2 * cd):
        raise CertificateError(f"Lemma 8.1(2) failed: k0={k0}, k0'={k0_p} out of [1, 2*cd)")
    if gcd(k0 * k0_p, P0 * ell) != 1:
        raise CertificateError("Lemma 8.1(2) failed: gcd(k0 * k0', P0 * ell) != 1")
    if parse_exact_int(bm["claimed_k0"], "base_m_pair.claimed_k0") != k0:
        raise CertificateError(f"claimed_k0 mismatch: actual {k0}")
    if parse_exact_int(bm["claimed_k0_prime"], "base_m_pair.claimed_k0_prime") != k0_p:
        raise CertificateError(f"claimed_k0_prime mismatch: actual {k0_p}")

    # Check f_i(a) == 1 (mod P0) for all residue classes a mod P0 (Lemma 8.1(3))
    # Since P0 = 6 (z0=3) or 30 (z0=5), checking all a in [0, P0-1] is exact and O(P0*d)!
    for a in range(P0):
        if poly_eval_mod(f1, a, P0) != 1 or poly_eval_mod(f2, a, P0) != 1:
            raise CertificateError(f"Lemma 8.1(3) failed: f_i({a}) != 1 (mod P0={P0})")

    # Check Lemma 8.1(4): f1 splits into d distinct simple roots mod ell, while f2 has ZERO roots mod ell!
    df1 = poly_deriv_int(f1)
    roots_f1 = [r for r in range(ell) if poly_eval_mod(f1, r, ell) == 0]
    if len(roots_f1) != d or any(poly_eval_mod(df1, r, ell) == 0 for r in roots_f1):
        raise CertificateError(f"Lemma 8.1(4) failed: f1 does not split into d={d} simple roots mod ell={ell}")
    if any(poly_eval_mod(f2, a, ell) == 0 for a in range(ell)):
        raise CertificateError(f"Lemma 8.1(4) failed: f2 has a root modulo ell={ell}")

    # 2. Lemma 11.1 Subgroup-Index Saturation Audit
    sat = payload["saturation_probe"]
    if not isinstance(sat, dict):
        raise CertificateError("saturation_probe must be an object")
    _require_keys(
        sat,
        required={"P_ok", "M_star", "K_idx", "claimed_L_idx", "claimed_M_tilde"},
        optional=set(),
        ctx="saturation_probe",
    )
    pok_raw = sat["P_ok"]
    if not isinstance(pok_raw, list) or not pok_raw:
        raise CertificateError("saturation_probe.P_ok must be a non-empty list of primes")
    P_ok = [parse_exact_int(x, f"P_ok[{i}]") for i, x in enumerate(pok_raw)]
    if len(set(P_ok)) != len(P_ok) or any(not is_prime_exact(q) for q in P_ok):
        raise CertificateError("saturation_probe.P_ok must be duplicate-free primes")

    M_star = parse_exact_int(sat["M_star"], "saturation_probe.M_star")
    K_idx = parse_exact_int(sat["K_idx"], "saturation_probe.K_idx")
    if M_star < 2 or K_idx < 2 or K_idx > 300:
        raise CertificateError("saturation_probe requires M_star >= 2 and 2 <= K_idx <= 300")
    if min(p1, p2) <= K_idx:
        raise CertificateError(f"Lemma 11.1 requires P^-(N)={min(p1, p2)} > K_idx={K_idx}")

    # Verify ell^{M_star} == 1 (mod N) for EVERY ell in P_ok!
    for q in P_ok:
        if pow(q, M_star, N) != 1:
            raise CertificateError(f"Lemma 11.1 annihilation failed: {q}^{{M_star}} != 1 (mod N)")

    L_idx = lcm_1_to_k(K_idx)
    M_tilde = M_star * L_idx
    if parse_exact_int(sat["claimed_L_idx"], "saturation_probe.claimed_L_idx") != L_idx:
        raise CertificateError(f"claimed_L_idx mismatch: actual {L_idx}")
    if parse_exact_int(sat["claimed_M_tilde"], "saturation_probe.claimed_M_tilde") != M_tilde:
        raise CertificateError(f"claimed_M_tilde mismatch: actual {M_tilde}")

    # Verify exact Lemma 11.1 conclusion: (p1 - 1) | M_tilde AND (p2 - 1) | M_tilde AND M_tilde even!
    if M_tilde % 2 != 0 or M_tilde % (p1 - 1) != 0 or M_tilde % (p2 - 1) != 0:
        raise CertificateError(
            f"Lemma 11.1 saturation conclusion failed: (p1-1)={p1-1} or (p2-1)={p2-1} does not divide M_tilde"
        )

    return {
        "N": N,
        "p1": p1,
        "p2": p2,
        "base_m_d": d,
        "base_m_m": m,
        "base_m_cd": cd,
        "base_m_ell": ell,
        "base_m_k0": k0,
        "base_m_k0_prime": k0_p,
        "P_ok_size": len(P_ok),
        "K_idx": K_idx,
        "L_idx_bits": L_idx.bit_length(),
        "M_tilde_bits": M_tilde.bit_length(),
    }


# ---------------------------------------------------------------------------
# Unified Entry Point
# ---------------------------------------------------------------------------


def verify_certificate_dict(cert: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cert, dict):
        raise CertificateError("Certificate must be a JSON object")
    _check_depth_and_types(cert)
    _require_keys(
        cert,
        required={"schema_version", "cert_id", "mode", "payload"},
        optional={"description"},
        ctx="certificate",
    )
    if cert["schema_version"] != SCHEMA_VERSION:
        raise CertificateError(
            f"Unsupported schema_version: {cert['schema_version']!r} (expected {SCHEMA_VERSION!r})"
        )
    if "description" in cert and not isinstance(cert["description"], str):
        raise CertificateError("description must be a string")
    cert_id = _require_str(cert["cert_id"], "cert_id")
    mode = _require_str(cert["mode"], "mode")
    if mode not in VALID_MODES:
        raise CertificateError(f"Unknown certificate mode: {mode!r}")
    payload = cert["payload"]
    if not isinstance(payload, dict):
        raise CertificateError("payload must be a JSON object")
    if "note" in payload and not isinstance(payload["note"], str):
        raise CertificateError("payload.note must be a string")

    try:
        if mode == "miller_multiquadratic_compat_cert":
            summary = _verify_mode1(payload)
        elif mode == "quadratic_infrastructure_wh_cert":
            summary = _verify_mode2(payload)
        elif mode == "fixedpoint_hnf_and_trypair_cert":
            summary = _verify_mode3(payload)
        else:
            summary = _verify_mode4(payload)
    except CertificateError:
        raise
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc

    return {
        "status": "VERIFIED",
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": mode,
        "summary": summary,
    }


def verify_certificate_json(raw: str) -> Dict[str, Any]:
    cert = load_strict_json(raw)
    return verify_certificate_dict(cert)


def verify_certificate_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return verify_certificate_json(p.read_text(encoding="utf-8"))
