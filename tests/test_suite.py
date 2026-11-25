"""Comprehensive 64-test suite for Paper 61 (DETFACT-COGENTIC-CERT-1.0):
- 1 aggregate replay of all 16 canonical certificates
- 3 raw-JSON zero-float / NaN / duplicate-key parser guards
- 60 parametrized live adversarial forgery attacks (12 schema/parser + 12 per mode),
  each asserting an exact regex match on the intended CertificateError branch.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detfact_certifier import (
    CertificateError,
    verify_certificate_dict,
    verify_certificate_file,
    verify_certificate_json,
)
from src.detfact_exact import is_prime_exact, miller_multiquadratic_roots

CERT_DIR = ROOT / "certificates"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((CERT_DIR / f"{name}.cert.json").read_text(encoding="utf-8"))


def _mod(base: Dict[str, Any], fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    c = copy.deepcopy(base)
    fn(c)
    return c


def test_all_16_canonical_certificates_verify() -> None:
    paths = sorted(CERT_DIR.glob("*.cert.json"))
    assert len(paths) == 16
    modes_seen = set()
    for p in paths:
        res = verify_certificate_file(p)
        assert res["status"] == "VERIFIED"
        modes_seen.add(res["mode"])
    assert len(modes_seen) == 4


def test_raw_json_rejects_ieee754_float() -> None:
    raw = (CERT_DIR / "m1_miller_split_n323.cert.json").read_text(encoding="utf-8")
    tampered = raw.replace('"N": 323', '"N": 323.0', 1)
    with pytest.raises(CertificateError, match="Floating-point constant rejected"):
        verify_certificate_json(tampered)


def test_raw_json_rejects_nan_constant() -> None:
    raw = (CERT_DIR / "m1_miller_split_n323.cert.json").read_text(encoding="utf-8")
    tampered = raw.replace('"N": 323', '"N": NaN', 1)
    with pytest.raises(CertificateError, match="Floating-point constant rejected"):
        verify_certificate_json(tampered)


def test_raw_json_rejects_duplicate_keys() -> None:
    raw = '{"schema_version":"DETFACT-COGENTIC-CERT-1.0","schema_version":"DETFACT-COGENTIC-CERT-1.0"}'
    with pytest.raises(CertificateError, match="Duplicate JSON key"):
        verify_certificate_json(raw)


def test_noncyclic_2sylow_splits_n3601_deterministically() -> None:
    """Regression test: N = 13 * 277 = 3601 has non-cyclic 2-Sylow subgroup (z_3^2 = -1)."""
    res = miller_multiquadratic_roots(3601, 276, 3)
    assert res["outcome"] == "SPLIT"
    d = int(res["divisor"])
    assert d in (13, 277) and 3601 % d == 0


def test_is_prime_exact_rejects_12base_sprp() -> None:
    """Regression test: Sorenson-Webster 12-base strong pseudoprime is rejected by 13-base MR."""
    sprp12 = 318665857834031151167461  # 399165290221 * 798330580441
    assert is_prime_exact(sprp12) is False


def _build_60_forgeries() -> List[Tuple[str, str, Dict[str, Any]]]:
    m1_s = _load("m1_miller_split_n323")
    m1_f = _load("m1_compat_family_p211_k3")
    m2 = _load("m2_wh_q71_gauss_d2")
    m3 = _load("m3_fp_hnf2_trypair_n323")
    m4 = _load("m4_basem_sat_n64507_d4")

    def _deep_nest(c: Dict[str, Any]) -> None:
        x: Any = 1
        for _ in range(20):
            x = [x]
        c["payload"]["N"] = x

    def _m1_sign_flip_biquad(c: Dict[str, Any]) -> None:
        # Negate one s_ell in family: keeps s_ell^2 == d_ell (mod p) STILL TRUE,
        # while BREAKING pairwise biquadratic compatibility s_l1 * s_l2 == eta12 * s_l3!
        p = c["payload"]["N"]
        c["payload"]["claimed_family"]["100"]["s"] = (-c["payload"]["claimed_family"]["100"]["s"]) % p

    def _m1_non_sqfree_d(c: Dict[str, Any]) -> None:
        c["payload"]["claimed_family"]["100"]["d"] = 4

    def _m2_break_biquad(c: Dict[str, Any]) -> None:
        q = c["payload"]["q_prime"]
        c["payload"]["family"]["10"]["s"] = (-c["payload"]["family"]["10"]["s"]) % q

    def _m3_singular_B(c: Dict[str, Any]) -> None:
        c["payload"]["trypair_annihilation_probe"]["pivot_matrix_B"] = [[2, 4], [1, 2]]

    def _m3_bad_adj(c: Dict[str, Any]) -> None:
        c["payload"]["trypair_annihilation_probe"]["claimed_adj_B"][0][0] += 1

    def _m4_f1_reducible_f2(c: Dict[str, Any]) -> None:
        # Flip constant term by +6? No, flip by +3 (mod 2 flips from 1 to 0 -> divisible by X in F_2[X]!)
        c["payload"]["base_m_pair"]["f1_coeffs"][0] += 3

    def _m4_f1_bad_eval_m(c: Dict[str, Any]) -> None:
        # Add 6*7 = 42 to c_0 -> keeps mod 2, mod 3, mod 7 identical, breaks f1(m) == 0 (mod N)!
        c["payload"]["base_m_pair"]["f1_coeffs"][0] += 42

    def _m4_f2_root_mod_ell(c: Dict[str, Any]) -> None:
        # Copy f1_coeffs into f2_coeffs: satisfies mod 2, mod 6, eval at m, but f2 now has roots mod ell!
        c["payload"]["base_m_pair"]["f2_coeffs"] = list(c["payload"]["base_m_pair"]["f1_coeffs"])
        c["payload"]["base_m_pair"]["claimed_k0_prime"] = c["payload"]["base_m_pair"]["claimed_k0"]

    def _m4_bad_M_star_annih(c: Dict[str, Any]) -> None:
        c["payload"]["saturation_probe"]["M_star"] = 2

    def _m4_insufficient_K_idx(c: Dict[str, Any]) -> None:
        c["payload"]["saturation_probe"]["K_idx"] = 301

    return [
        # --- 12 Schema & Parser Forgeries ---
        ("S01_wrong_schema", "Unsupported schema_version", _mod(m1_s, lambda c: c.__setitem__("schema_version", "BAD-1.0"))),
        ("S02_unknown_mode", "Unknown certificate mode", _mod(m1_s, lambda c: c.__setitem__("mode", "bogus"))),
        ("S03_missing_payload", "missing required keys", _mod(m1_s, lambda c: c.pop("payload"))),
        ("S04_extra_top_key", "unexpected extra keys", _mod(m1_s, lambda c: c.__setitem__("extra", 1))),
        ("S05_empty_cert_id", "expected non-empty string", _mod(m1_s, lambda c: c.__setitem__("cert_id", " "))),
        ("S06_bool_as_int", "boolean not permitted", _mod(m1_s, lambda c: c["payload"].__setitem__("N", True))),
        ("S07_float_in_dict", "Floating-point value", _mod(m1_s, lambda c: c["payload"].__setitem__("N", 323.5))),
        ("S08_unicode_digit_str", "invalid exact integer", _mod(m1_s, lambda c: c["payload"].__setitem__("N", "٣٢٣"))),
        ("S09_excessive_nesting", "nesting depth exceeded", _mod(m1_s, _deep_nest)),
        ("S10_extra_payload_key", "unexpected extra keys", _mod(m1_s, lambda c: c["payload"].__setitem__("injected", 9))),
        ("S11_payload_not_dict", "payload must be a JSON object", _mod(m1_s, lambda c: c.__setitem__("payload", []))),
        ("S12_empty_str_int", "empty integer string", _mod(m1_s, lambda c: c["payload"].__setitem__("N", ""))),

        # --- 12 Mode 1 (Miller 2-Sylow & Multiquadratic Compatibility) Forgeries ---
        ("M1_01_even_N", "requires odd N >= 9", _mod(m1_s, lambda c: c["payload"].__setitem__("N", 324))),
        ("M1_02_odd_M_tilde", "requires even M_tilde", _mod(m1_s, lambda c: c["payload"].__setitem__("M_tilde", 145))),
        ("M1_03_K_out_of_range", "requires 2 <= K <= 5", _mod(m1_s, lambda c: c["payload"].__setitem__("K", 1))),
        ("M1_04_invalid_outcome", "Invalid Mode 1 claimed_outcome", _mod(m1_s, lambda c: c["payload"].__setitem__("claimed_outcome", "MAYBE"))),
        ("M1_05_outcome_mismatch", "claimed_outcome mismatch", _mod(m1_s, lambda c: c["payload"].__setitem__("claimed_outcome", "COMPATIBLE_FAMILY"))),
        ("M1_06_split_missing_div", "SPLIT outcome requires claimed_divisor", _mod(m1_s, lambda c: c["payload"].pop("claimed_divisor"))),
        ("M1_07_forged_divisor", "not a nontrivial divisor", _mod(m1_s, lambda c: c["payload"].__setitem__("claimed_divisor", 15))),
        ("M1_08_trivial_divisor_1", "not a nontrivial divisor", _mod(m1_s, lambda c: c["payload"].__setitem__("claimed_divisor", 1))),
        ("M1_09_forged_j_star", "claimed_j_star mismatch", _mod(m1_f, lambda c: c["payload"].__setitem__("claimed_j_star", (c["payload"]["claimed_j_star"] + 1) % 4))),
        ("M1_10_forged_t_star", "claimed_t_star mismatch", _mod(m1_f, lambda c: c["payload"].__setitem__("claimed_t_star", 99))),
        ("M1_11_sign_flip_breaks_biquad", "failed pairwise biquadratic compatibility", _mod(m1_f, _m1_sign_flip_biquad)),
        ("M1_12_non_squarefree_d", "failed pairwise biquadratic compatibility", _mod(m1_f, _m1_non_sqfree_d)),

        # --- 12 Mode 2 (Gauss Reduction Cycles, Walsh-Hadamard & QNR) Forgeries ---
        ("M2_01_composite_q_prime", "requires odd prime q_prime", _mod(m2, lambda c: c["payload"].__setitem__("q_prime", 77))),
        ("M2_02_qnr_is_residue", "NOT a quadratic nonresidue", _mod(m2, lambda c: c["payload"].__setitem__("claimed_qnr_z", 4))),
        ("M2_03_qnr_out_of_range", "out of range", _mod(m2, lambda c: c["payload"].__setitem__("claimed_qnr_z", 71))),
        ("M2_04_biquad_sign_flip", "failed biquadratic compatibility", _mod(m2, _m2_break_biquad)),
        ("M2_05_non_squarefree_gauss_d", "must be squarefree in \\(1, 10000\\]", _mod(m2, lambda c: c["payload"].__setitem__("gauss_real_d", 4))),
        ("M2_06_forged_gauss_period", "claimed_gauss_period mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_gauss_period", 9))),
        ("M2_07_forged_unit_x", "claimed_unit_x mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_unit_x", 999))),
        ("M2_08_forged_unit_y", "claimed_unit_y mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_unit_y", 999))),
        ("M2_09_forged_unit_norm", "claimed_unit_norm mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_unit_norm", 1))),
        ("M2_10_Y_contains_qnr", "not identically \\+1 on \\[1, Y=", _mod(m2, lambda c: c["payload"].__setitem__("Y", 7))),
        ("M2_11_forged_smooth_Psi", "claimed_smooth_Psi mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_smooth_Psi", 999))),
        ("M2_12_forged_max_S_ell", "claimed_max_S_ell mismatch", _mod(m2, lambda c: c["payload"].__setitem__("claimed_max_S_ell", 999))),
        ("M2_13_malformed_family_keys", "family keys must be exact", _mod(m2, lambda c: (c["payload"]["family"].pop("10"), c["payload"]["family"].__setitem__("XX", {"d": 2, "s": 9})))),
        ("M2_14_ramified_d_outside_PK", "must be supported on", _mod(m2, lambda c: c["payload"]["family"]["10"].__setitem__("d", 5))),

        # --- 14 Mode 3 (Fixed-Point ApproxLog/Sqrt, ModHNF & TryPair) Forgeries ---
        ("M3_01_forged_approx_log", "approx_log_probe.claimed_num mismatch", _mod(m3, lambda c: c["payload"]["approx_log_probe"].__setitem__("claimed_num", 1))),
        ("M3_02_forged_approx_sqrt", "approx_sqrt_probe.claimed_num mismatch", _mod(m3, lambda c: c["payload"]["approx_sqrt_probe"].__setitem__("claimed_num", 1))),
        ("M3_03_hnf_dim_too_large", "ModHNF requires 1 <= m <= 4", _mod(m3, lambda c: c["payload"]["mod_hnf_probe"].__setitem__("m", 5))),
        ("M3_04_forged_hnf_matrix", "mod_hnf_probe.claimed_H mismatch", _mod(m3, lambda c: c["payload"]["mod_hnf_probe"]["claimed_H"][0].__setitem__(0, 1))),
        ("M3_05_singular_B", "must be nonsingular", _mod(m3, _m3_singular_B)),
        ("M3_06_forged_det_B", "claimed_det_B mismatch", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("claimed_det_B", 99))),
        ("M3_07_forged_adj_B", "claimed_adj_B mismatch", _mod(m3, _m3_bad_adj)),
        ("M3_08_E0_ne_abs_det", "must equal \\|det\\(B\\)\\|", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("E0", 99))),
        ("M3_09_bad_sign_s1", "signs s1, s2 must lie in", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("s1", 2))),
        ("M3_10_bad_field1_cong", "field-1 localization congruence", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("s1", -c["payload"]["trypair_annihilation_probe"]["s1"]))),
        ("M3_11_bad_field2_cong", "field-2 localization congruence", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("s2", -c["payload"]["trypair_annihilation_probe"]["s2"]))),
        ("M3_12_forged_E_ell", "claimed_E_ell mismatch", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"].__setitem__("claimed_E_ell", 4))),
        ("M3_13_nonlist_hnf_col", "columns\\[0\\] must be a list", _mod(m3, lambda c: c["payload"]["mod_hnf_probe"]["columns"].__setitem__(0, 42))),
        ("M3_14_nonlist_adj_row", "claimed_adj_B\\[0\\] must be a list", _mod(m3, lambda c: c["payload"]["trypair_annihilation_probe"]["claimed_adj_B"].__setitem__(0, 42))),

        # --- 14 Mode 4 (Preconditioned Base-m Pair, Saturation & Factoring) Forgeries ---
        ("M4_01_wrong_product_N", "Invalid nontrivial factorization", _mod(m4, lambda c: c["payload"].__setitem__("factor_p1", 249))),
        ("M4_02_perfect_square_N", "non-perfect-power semiprime", _mod(m4, lambda c: (c["payload"].__setitem__("N", 251 * 251), c["payload"].__setitem__("factor_p2", 251)))),
        ("M4_03_bad_base_m_bracket", "base-m bracket", _mod(m4, lambda c: c["payload"]["base_m_pair"].__setitem__("m", 17))),
        ("M4_04_cd_mismatch", "leading coefficient cd condition", _mod(m4, lambda c: c["payload"]["base_m_pair"]["f1_coeffs"].__setitem__(-1, 3))),
        ("M4_05_f1_reducible_f2", "f1 mod 2 is not irreducible", _mod(m4, _m4_f1_reducible_f2)),
        ("M4_06_f1_bad_eval_m", "f_i\\(m\\) != 0", _mod(m4, _m4_f1_bad_eval_m)),
        ("M4_07_forged_k0", "claimed_k0 mismatch", _mod(m4, lambda c: c["payload"]["base_m_pair"].__setitem__("claimed_k0", 1))),
        ("M4_08_f2_has_root_mod_ell", "f2 has a root modulo ell", _mod(m4, _m4_f2_root_mod_ell)),
        ("M4_09_nonprime_in_Pok", "duplicate-free primes", _mod(m4, lambda c: c["payload"]["saturation_probe"]["P_ok"].append(15))),
        ("M4_10_M_star_fails_annih", "Lemma 11.1 annihilation failed", _mod(m4, _m4_bad_M_star_annih)),
        ("M4_11_K_idx_out_of_cap", "2 <= K_idx <= 300", _mod(m4, _m4_insufficient_K_idx)),
        ("M4_12_forged_L_idx", "claimed_L_idx mismatch", _mod(m4, lambda c: c["payload"]["saturation_probe"].__setitem__("claimed_L_idx", 12345))),
        ("M4_13_pmin_le_K_idx", "requires P\\^-\\(N\\)", _mod(m4, lambda c: c["payload"]["saturation_probe"].__setitem__("K_idx", 251))),
        ("M4_14_p1_minus_1_not_div_M_tilde", "saturation conclusion failed", _mod(m4, lambda c: (c["payload"]["saturation_probe"].__setitem__("P_ok", [2]), c["payload"]["saturation_probe"].__setitem__("M_star", 400), c["payload"]["saturation_probe"].__setitem__("K_idx", 2), c["payload"]["saturation_probe"].__setitem__("claimed_L_idx", 2), c["payload"]["saturation_probe"].__setitem__("claimed_M_tilde", 800)))),
    ]


@pytest.mark.parametrize("attack_id,expected_regex,forged_cert", _build_60_forgeries())
def test_adversarial_forgery_rejected(
    attack_id: str,
    expected_regex: str,
    forged_cert: Dict[str, Any],
) -> None:
    with pytest.raises(CertificateError, match=expected_regex):
        verify_certificate_dict(forged_cert)
