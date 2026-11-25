# Constructive Lean 4 Kernel and Exact Zero-Float Certificates for Google Research Cogentic's Deterministic Subexponential Factoring and QNR Breakthrough

**Author:** Lluis Eriksson (Independent Researcher)
**License:** MIT (code, Lean 4 kernel, certificates) / CC-BY-4.0 (manuscript & figures)
**Target Breakthrough:** Google Research Cogentic (Nov. 25, 2026), *Unconditional Deterministic Subexponential Integer Factoring and Quadratic Nonresidue Construction* ($O(\exp(3(\ln N)^{3/4}))$ factoring & $L_p[1/2, O(1)]$ QNR).

## Repository Architecture
- `lean/CogenticDetFactoring.lean` + `lean/AuditAxioms.lean`: 18 standalone constructive Lean 4.35.0 theorems (0 `sorry`, 0 `axiom`, 0 `Classical.choice`, 0 warnings; 3 zero-axiom `none` + 15 `[propext, Quot.sound]`).
- `src/detfact_exact.py` + `src/detfact_certifier.py`: Strictly zero-float (`int` / `fractions.Fraction`) 4-mode verifier (`DETFACT-COGENTIC-CERT-1.0`).
- `certificates/`: 16 canonical exact certificates across all 4 modes (Miller 2-Sylow splitting & 105-pair biquadratic compatibility, Gauss reduction cycles & Walsh-Hadamard positivity + QNRs, fixed-point ApproxLog/ApproxSqrt + ModHNF + TryPair localization annihilation, and Lemma 8.1 base-$m$ pairs + Lemma 11.1 subgroup-index saturation).
- `tests/test_suite.py`: 72 pytest checks (16-cert replay + 3 raw-JSON parser guards + 2 soundness regressions + 66 live adversarial forgery attacks with exact rejection-branch assertions).
- `scripts/verify_all.py`: Unified 5-gate verifier (Lean build/axioms, zero-float AST audit, 16-cert replay, 72/72 pytest, 40-file SHA-256 manifest).
- `paper/main.pdf`: 4-page publication manuscript with 3 300-DPI figures and worked hand-verifiable witnesses.

## Quick Verification
```bash
python scripts/verify_all.py
```
