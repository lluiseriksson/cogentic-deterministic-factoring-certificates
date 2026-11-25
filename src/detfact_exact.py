"""Pure-stdlib exact integer and rational arithmetic kernel for Paper 61:
Unconditional Deterministic Subexponential Integer Factoring and Quadratic Nonresidue
Construction (Cogentic 2026: O(exp(3 (ln N)^(3/4))) & L_p[1/2, O(1)]).

Strictly zero floating-point arithmetic: uses only Python arbitrary-precision `int`
and `fractions.Fraction`.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Strict Parsing (Zero-Float, Strict ASCII Digits)
# ---------------------------------------------------------------------------


def parse_exact_int(val: object, field: str) -> int:
    """Strictly parse an exact integer (rejecting bools, floats, non-ASCII digits)."""
    if isinstance(val, bool):
        raise ValueError(f"{field}: boolean not permitted where integer expected")
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError(f"{field}: empty integer string")
        sign = 1
        if s[0] == "-":
            sign = -1
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]
        if not s or not (s.isascii() and s.isdigit()):
            raise ValueError(f"{field}: invalid exact integer string {val!r}")
        return sign * int(s)
    raise ValueError(f"{field}: expected exact int or integer string, got {type(val).__name__}")


def parse_exact_fraction(val: object, field: str) -> Fraction:
    """Strictly parse an exact rational 'num' or 'num/den' (rejecting floats/decimals)."""
    if isinstance(val, bool):
        raise ValueError(f"{field}: boolean not permitted where rational expected")
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError(f"{field}: empty rational string")
        if "." in s or "e" in s or "E" in s:
            raise ValueError(f"{field}: decimal/scientific notation rejected in exact rational {val!r}")
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                raise ValueError(f"{field}: malformed rational {val!r}")
            num = parse_exact_int(parts[0], f"{field}.num")
            den = parse_exact_int(parts[1], f"{field}.den")
            if den <= 0:
                raise ValueError(f"{field}: rational denominator must be positive, got {den}")
            return Fraction(num, den)
        return Fraction(parse_exact_int(s, field), 1)
    raise ValueError(f"{field}: expected exact rational string or int, got {type(val).__name__}")


def fmt_frac(q: Fraction) -> str:
    return f"{q.numerator}" if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------------------
# Exact Number-Theoretic Primitives
# ---------------------------------------------------------------------------


DETERMINISTIC_MR_LIMIT = 3_317_044_064_679_887_385_961_981  # Sorenson-Webster 13-base limit


def is_prime_exact(n: int) -> bool:
    """Deterministic exact primality test (13-base Miller-Rabin proven exact for n < 3.317e24)."""
    if n < 2:
        return False
    if n >= DETERMINISTIC_MR_LIMIT:
        raise ValueError(f"n={n} exceeds proven 13-base deterministic Miller-Rabin limit {DETERMINISTIC_MR_LIMIT}")
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 41 * 41:
        return True
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def first_k_primes(k: int) -> List[int]:
    primes: List[int] = []
    cand = 2
    while len(primes) < k:
        if is_prime_exact(cand):
            primes.append(cand)
        cand += 1 if cand == 2 else 2
    return primes


def is_squarefree(d: int) -> bool:
    if d == 0:
        return False
    x = abs(d)
    p = 2
    while p * p <= x:
        if x % (p * p) == 0:
            return False
        if x % p == 0:
            x //= p
        p += 1 if p == 2 else 2
    return True


def fundamental_discriminant(d: int) -> int:
    """Fundamental discriminant Delta_F of F = Q(sqrt(d)) for squarefree d != 1."""
    if d == 1 or not is_squarefree(d):
        raise ValueError(f"d={d} must be squarefree and != 1")
    return d if (d % 4 == 1) else 4 * d


def legendre_symbol(a: int, p: int) -> int:
    """Exact Legendre symbol (a/p) for odd prime p via Euler's criterion."""
    if p < 3 or p % 2 == 0 or not is_prime_exact(p):
        raise ValueError(f"legendre_symbol requires odd prime p, got {p}")
    a_mod = a % p
    if a_mod == 0:
        return 0
    r = pow(a_mod, (p - 1) // 2, p)
    return 1 if r == 1 else -1


def jacobi_symbol(a: int, n: int) -> int:
    """Exact Jacobi symbol (a/n) for odd positive n."""
    if n <= 0 or n % 2 == 0:
        raise ValueError(f"jacobi_symbol requires odd positive n, got {n}")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm returning (g, u, v) with g >= 0 and u*a + v*b = g."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def mod_inv(a: int, m: int) -> int:
    g, u, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} is not invertible modulo {m} (gcd={g})")
    return u % m


def lcm_int(a: int, b: int) -> int:
    return (a // gcd(a, b)) * b


def prime_factorization_small(n: int) -> List[Tuple[int, int]]:
    if n < 1:
        raise ValueError(f"Expected positive integer, got {n}")
    out: List[Tuple[int, int]] = []
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            e = 0
            while x % d == 0:
                x //= d
                e += 1
            out.append((d, e))
        d += 1 if d == 2 else 2
    if x > 1:
        out.append((x, 1))
    return out


# ---------------------------------------------------------------------------
# Section 2.3: Exact Deterministic Fixed-Point Subroutines & ModHNF
# ---------------------------------------------------------------------------


def approx_log_exact(u: int, v: int, M: int) -> Tuple[int, int]:
    """Cogentic Lemma 2.2(1) exact integer-only ApproxLog(u, v, M).

    Returns (num, M) representing dyadic rational num * 2^(-M) with
    |ln(u/v) - num * 2^(-M)| < 2^(-M).
    """
    if u < 1 or v < 1 or M < 4:
        raise ValueError("ApproxLog requires u >= 1, v >= 1, M >= 4")
    if u == v:
        return 0, M
    if u < v:
        num, _ = approx_log_exact(v, u, M)
        return -num, M
    # Find unique k >= 0 with 2^k v <= u < 2^(k+1) v
    k = 0
    v_shift = v
    while (v_shift << 1) <= u:
        v_shift <<= 1
        k += 1
    # J = M + floor(log2(k+4)) + 1, M' = M + floor(log2(8*J*(k+4))) + 1
    J = M + ((k + 4).bit_length() - 1) + 1
    M_prime = M + ((8 * J * (k + 4)).bit_length() - 1) + 1

    def s_int(p: int, q: int) -> int:
        if p == 0:
            return 0
        t = (p << (M_prime + 1)) // q
        total = t  # j = 0 -> t // 1
        p2 = p * p
        q2 = q * q
        for j in range(1, J):
            t = (t * p2) // q2
            total += t // (2 * j + 1)
        return total

    s1 = s_int(1, 3)
    s2 = s_int(u - v_shift, u + v_shift)
    s_tot = k * s1 + s2
    # Round s_tot * 2^(M - M_prime) to nearest integer: floor((2*s_tot + 2^(M'-M)) / 2^(M'-M+1))
    shift = M_prime - M
    rounded = (s_tot + (1 << (shift - 1))) >> shift
    return rounded, M


def approx_sqrt_exact(u: int, v: int, M: int) -> Tuple[int, int]:
    """Cogentic Lemma 2.2(3) exact integer-only ApproxSqrt(u, v, M).

    Returns (num, M) representing num * 2^(-M) with |sqrt(u/v) - num * 2^(-M)| < 2^(-M).
    """
    if u < 1 or v < 1 or M < 2:
        raise ValueError("ApproxSqrt requires u >= 1, v >= 1, M >= 2")
    n0 = (u << (2 * M + 4)) // v
    t = isqrt(n0)
    num = (t + 2) >> 2
    return num, M


def mod_hnf_exact(columns: Sequence[Sequence[int]], m: int, d_mod: int) -> List[List[int]]:
    """Cogentic Lemma 2.2(4) exact deterministic ModHNF in Z^m (1 <= m <= 4).

    Returns upper-triangular column HNF matrix H (list of m columns, each length m)
    spanning sum Z*a_j + d_mod Z^m, with h_{ii} >= 1, h_{ij} = 0 for i > j,
    and 0 <= h_{ij} < h_{ii} for i < j.
    """
    if not (1 <= m <= 4) or d_mod < 1:
        raise ValueError(f"ModHNF requires 1 <= m <= 4 and d_mod >= 1, got m={m}, d_mod={d_mod}")
    active: List[List[int]] = []
    for col in columns:
        if len(col) != m:
            raise ValueError(f"Column length {len(col)} != m={m}")
        active.append([x % d_mod for x in col])
    for i in range(m):
        e_col = [0] * m
        e_col[i] = d_mod
        active.append(e_col)

    pivots: List[Optional[List[int]]] = [None] * m
    for row in range(m - 1, -1, -1):
        residual: List[List[int]] = []
        piv: Optional[List[int]] = None
        for col in active:
            if col[row] == 0:
                residual.append(col)
                continue
            if piv is None:
                c = list(col)
                if c[row] < 0:
                    c = [-x for x in c]
                for r in range(row):
                    c[r] %= d_mod
                piv = c
            else:
                pi = piv[row]
                ci = col[row]
                g, u, v = egcd(pi, ci)
                a_p = pi // g
                b_p = ci // g
                new_piv = [0] * m
                new_col = [0] * m
                for r in range(m):
                    new_piv[r] = u * piv[r] + v * col[r]
                    new_col[r] = -b_p * piv[r] + a_p * col[r]
                for r in range(row):
                    new_piv[r] %= d_mod
                    new_col[r] %= d_mod
                piv = new_piv
                residual.append(new_col)
        assert piv is not None and piv[row] >= 1 and d_mod % piv[row] == 0
        pivots[row] = piv
        active = residual

    H: List[List[int]] = [list(p) for p in pivots if p is not None]
    # Reduce upper-triangular off-diagonals: for i = m-1..0, j = i+1..m-1
    for i in range(m - 1, -1, -1):
        for j in range(i + 1, m):
            q = H[j][i] // H[i][i]
            for r in range(i + 1):
                H[j][r] -= q * H[i][r]
    return H


# ---------------------------------------------------------------------------
# Section 3: Miller 2-Sylow Order Test & Compatible Multiquadratic Square Roots
# ---------------------------------------------------------------------------


def two_adic_decomp(m_even: int) -> Tuple[int, int]:
    """Write positive even m_even = 2^e * u_odd with u_odd odd."""
    if m_even < 2 or m_even % 2 != 0:
        raise ValueError(f"Expected positive even exponent, got {m_even}")
    e = 0
    u = m_even
    while u % 2 == 0:
        u //= 2
        e += 1
    return e, u


def miller_multiquadratic_roots(
    N: int,
    M_tilde: int,
    K: int,
) -> Dict[str, object]:
    """Run Cogentic Section 3 deterministic Miller 2-Sylow order test on q_0=-1, q_1,...,q_K.

    Either returns a verified nontrivial divisor 1 < g < N, OR extracts
    K squarefree integers d^(1),...,d^(K) and compatible modular square roots
    (d_ell, s_ell) for all 2^K - 1 nonzero vectors ell in F_2^K \\ {0}
    satisfying s_ell^2 == d_ell (mod N) and s_{l1}*s_{l2} == eta_{12}*s_{l1+l2} (mod N).
    """
    if N < 3 or N % 2 == 0:
        raise ValueError("N must be an odd integer >= 3")
    if K < 1 or K > 6:
        raise ValueError("K must lie in [1, 6] for exact multiquadratic certificate enumeration")
    e_max, u_odd = two_adic_decomp(M_tilde)
    qs = [-1] + first_k_primes(K)

    # Check coprimality
    for qv in qs:
        if qv > 0:
            g = gcd(qv, N)
            if 1 < g < N:
                return {"outcome": "SPLIT", "divisor": g, "reason": f"gcd({qv}, N)={g}"}
            if g == N:
                raise ValueError(f"Prime {qv} equals N")

    # Compute 2-power orders of z_j = q_j^{u_odd} mod N
    z_vals: List[int] = []
    orders_exp: List[int] = []
    for qv in qs:
        z = pow(qv % N, u_odd, N)
        cur = z
        t = 0
        while cur != 1:
            nxt = (cur * cur) % N
            if nxt == 1:
                if cur != N - 1:
                    g = gcd(cur - 1, N)
                    if 1 < g < N:
                        return {"outcome": "SPLIT", "divisor": g, "reason": f"Miller sqrt(-1/+1) split on q={qv}"}
            cur = nxt
            t += 1
            if t > e_max:
                raise ValueError(f"M_tilde does not annihilate q={qv} modulo N")
        z_vals.append(z)
        orders_exp.append(t)

    # Select j* maximizing 2-power order t_{j*}
    j_star = max(range(K + 1), key=lambda j: orders_exp[j])
    t_star = orders_exp[j_star]
    if t_star > 16:
        raise ValueError(f"2-Sylow exponent t_star={t_star} exceeds enumeration cap 16")
    zeta = z_vals[j_star]

    # Check if all z_j lie in <zeta>; find exact discrete log c_j in [0, 2^t_star - 1] with z_j == zeta^{c_j}
    zeta_pow_map: Dict[int, int] = {}
    cur = 1
    for c in range(1 << t_star):
        zeta_pow_map[cur] = c
        cur = (cur * zeta) % N

    # Extract square roots for remaining K elements i_1 < ... < i_K
    rem_indices = [j for j in range(K + 1) if j != j_star]
    d_base: List[int] = []
    s_base: List[int] = []
    deltas: List[int] = []

    q_star = qs[j_star]
    for j in rem_indices:
        if z_vals[j] not in zeta_pow_map:
            # Non-cyclic 2-Sylow subgroup: find exact k >= 0 with u = z_j^{2^k} not in <zeta>
            # while u^2 = zeta^{2*m} in <zeta>. Then w = u * zeta^{-m} satisfies w^2 == 1 (mod N)
            # and w not in {1, -1} (since both 1 and -1 lie in <zeta> whenever t_star >= 1),
            # deterministically yielding a nontrivial divisor gcd(w - 1, N) in <= t_star steps!
            u = z_vals[j]
            for _ in range(t_star + 1):
                u_sq = (u * u) % N
                if u_sq in zeta_pow_map:
                    c_sq = zeta_pow_map[u_sq]
                    # Since ord(u) <= 2^{t_star}, ord(u^2) <= 2^{t_star - 1}, so c_sq is even!
                    if c_sq % 2 != 0:
                        raise ValueError("Internal 2-Sylow maximality invariant violated")
                    m_half = c_sq // 2
                    inv_z = pow(zeta, (1 << t_star) - m_half, N) if m_half > 0 else 1
                    w = (u * inv_z) % N
                    g = gcd(w - 1, N)
                    if 1 < g < N:
                        return {
                            "outcome": "SPLIT",
                            "divisor": g,
                            "reason": f"Non-cyclic 2-Sylow ratio split on q={qs[j]}",
                        }
                    raise ValueError(f"Non-cyclic 2-Sylow element failed to split N={N}")
                u = u_sq
            raise ValueError(f"2-Sylow squaring exceeded bound t_star={t_star}")
        c_j = zeta_pow_map[z_vals[j]]
        delta_j = c_j % 2  # parity of c_j
        # q_j * q_{j*}^{delta_j} has EVEN exponent in <zeta> and odd power u_odd -> exact modular sqrt!
        # Wait: z_j * zeta^{delta_j} = zeta^{c_j + delta_j} has even exponent 2*h_j!
        # Specifically: (q_j * q_{j*}^{delta_j})^{u_odd} = zeta^{c_j + delta_j}
        # Let u_odd = 2*w + 1 => (q_j * q_{j*}^{delta_j}) = (q_j * q_{j*}^{delta_j})^{-2w} * zeta^{c_j + delta_j}
        # Wait: (q_j * q_{j*}^{delta_j})^{1} = ((q_j * q_{j*}^{delta_j})^{(u_odd + 1)//2} * zeta^{-(c_j + delta_j)//2})^2?
        # Let's check: LHS^2 = (q_j*q_*^d)^{u_odd + 1} * zeta^{-(c_j + d)} = (q_j*q_*^d) * (z_j*zeta^d) * zeta^{-(c_j+d)} = q_j*q_*^d !!
        d_r = qs[j] * (q_star if delta_j == 1 else 1)
        half_u = (u_odd + 1) // 2
        half_c = (c_j + delta_j) // 2
        # Inverse of zeta^{half_c} mod N: zeta^{2^t_star - half_c} mod N
        inv_zeta_half = pow(zeta, (1 << t_star) - (half_c % (1 << t_star)), N) if t_star > 0 else 1
        s_r = (pow(d_r % N, half_u, N) * inv_zeta_half) % N
        assert (s_r * s_r) % N == d_r % N
        d_base.append(d_r)
        s_base.append(s_r)
        deltas.append(delta_j)

    # Construct all 2^K - 1 compatible pairs (d_ell, s_ell) per Cogentic Sec. 3 Eq. (p. 14-15)
    q_star_inv = mod_inv(q_star % N, N)
    family: Dict[str, Dict[str, int]] = {}
    for mask in range(1, 1 << K):
        bits = [(mask >> r) & 1 for r in range(K)]
        w_ell = sum(deltas[r] for r in range(K) if bits[r])
        k_ell = w_ell // 2
        eps_ell = w_ell % 2
        d_ell = q_star if eps_ell == 1 else 1
        s_ell = pow(q_star_inv, k_ell, N)
        for r in range(K):
            if bits[r]:
                d_ell *= qs[rem_indices[r]]
                s_ell = (s_ell * s_base[r]) % N
        assert (s_ell * s_ell) % N == d_ell % N
        key = "".join(str(b) for b in bits)
        family[key] = {"d": d_ell, "s": s_ell}

    return {
        "outcome": "COMPATIBLE_FAMILY",
        "j_star": j_star,
        "q_star": q_star,
        "t_star": t_star,
        "deltas": deltas,
        "d_base": d_base,
        "s_base": s_base,
        "family": family,
    }


def check_biquadratic_compatibility(
    N: int,
    family: Dict[str, Dict[str, int]],
    K: int,
) -> bool:
    """Verify exact pairwise biquadratic compatibility s_{l1}*s_{l2} == eta_{12}*s_{l1+l2} (mod N)
    for EVERY unordered pair of distinct nonzero vectors l1, l2 in F_2^K \\ {0}.
    """
    keys = sorted(family.keys())
    if len(keys) != (1 << K) - 1:
        return False
    # Check each individual s_ell^2 == d_ell (mod N) and d_ell squarefree != 1
    seen_d = set()
    for k in keys:
        d_val = family[k]["d"]
        s_val = family[k]["s"]
        if d_val == 1 or not is_squarefree(d_val) or d_val in seen_d:
            return False
        seen_d.add(d_val)
        if gcd(s_val, N) != 1 or (s_val * s_val - d_val) % N != 0:
            return False
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            k3 = "".join(str(int(a) ^ int(b)) for a, b in zip(k1, k2))
            d1, s1 = family[k1]["d"], family[k1]["s"]
            d2, s2 = family[k2]["d"], family[k2]["s"]
            d3, s3 = family[k3]["d"], family[k3]["s"]
            g = gcd(abs(d1), abs(d2))
            eta12 = -g if (d1 < 0 and d2 < 0) else g
            if d1 * d2 != eta12 * eta12 * d3:
                return False
            if (s1 * s2 - eta12 * s3) % N != 0:
                return False
    return True


# ---------------------------------------------------------------------------
# Section 4 & 5: Real Quadratic Gauss Reduction Cycle & Walsh-Hadamard Sums
# ---------------------------------------------------------------------------


def gauss_reduction_cycle_real(d: int) -> Dict[str, object]:
    """Exact Cogentic Lemma 4.2(1) Gauss reduction cycle in real quadratic field F = Q(sqrt(d)) (d > 1 squarefree).

    Returns exact period L, reduced pairs (a_k, b_k), and exact fundamental unit
    epsilon_d = (x_u + y_u * sqrt(D)) / 2 satisfying x_u^2 - D * y_u^2 = +-4!
    """
    if d <= 1 or not is_squarefree(d):
        raise ValueError(f"Expected squarefree d > 1, got {d}")
    D = fundamental_discriminant(d)
    fD = isqrt(D)
    a0 = 1
    b0 = fD if (fD % 2 == D % 2) else (fD - 1)

    # Track eta_k = (p_k + q_k sqrt(D)) / 2 (Wait: inverse tracking epsilon = eta_L^{-1} directly!)
    # From Cogentic p. 16-17: eta_{k+1} = eta_k * (-b_k + sqrt(D)) / (2 * a_k)
    # Hence eta_{k+1}^{-1} = eta_k^{-1} * (2 * a_k) / (-b_k + sqrt(D))
    #   = eta_k^{-1} * (2 * a_k * (b_k + sqrt(D))) / (D - b_k^2)
    #   = eta_k^{-1} * (b_k + sqrt(D)) / (2 * a_{k+1})  since D - b_k^2 = 4 * a_k * a_{k+1}!
    # Keep inv_eta = (X + Y*sqrt(D)) / 2 as exact Fraction pair (or exact ints — actually ALWAYS integers in O_F!!
    # Wait: eta_k = I(a_k, b_k) generator, so eta_k * O_F = I(a_k, b_k) => eta_L^{-1} = epsilon_d in O_F!
    # Intermediate inv_eta_k has denominator a_k in Q(sqrt(D)), so exact Fraction keeps zero floats!)
    ux = Fraction(1, 1)
    uy = Fraction(0, 1)

    ak, bk = a0, b0
    cycle: List[Tuple[int, int]] = []
    while True:
        cycle.append((ak, bk))
        a_next = (D - bk * bk) // (4 * ak)
        r = (fD + bk) % (2 * a_next)
        b_next = fD - r
        # Multiply (ux + uy*sqrt(D)) by (bk + sqrt(D)) / (2 * a_next)
        # (ux + uy*sqrt(D)) * (bk + sqrt(D)) = (ux*bk + uy*D) + (ux + uy*bk)*sqrt(D)
        n_ux = (ux * bk + uy * D) / (2 * a_next)
        n_uy = (ux + uy * bk) / (2 * a_next)
        ux, uy = n_ux, n_uy
        ak, bk = a_next, b_next
        if (ak, bk) == (a0, b0):
            break

    # Fundamental unit epsilon = ux + uy*sqrt(D) = (2*ux + (2*uy)*sqrt(D)) / 2
    x_int = (2 * ux)
    y_int = (2 * uy)
    assert x_int.denominator == 1 and y_int.denominator == 1
    xi, yi = x_int.numerator, y_int.numerator
    norm4 = xi * xi - D * yi * yi
    assert abs(norm4) == 4, f"Unit norm check failed: x^2 - D y^2 = {norm4}"
    return {
        "d": d,
        "D": D,
        "period_L": len(cycle),
        "reduced_pairs": cycle,
        "unit_x": xi,
        "unit_y": yi,
        "unit_norm": norm4 // 4,
    }


def walsh_hadamard_norm_sums_exact(
    q_mod: int,
    family: Dict[str, Dict[str, int]],
    K: int,
    X: int,
    Y: int,
) -> Dict[str, object]:
    """Compute exact Cogentic Lemma 5.1(2)-(3) norm-summed coefficients b_ell(n)
    and Walsh-Hadamard positivity check across all 1 <= n <= X coprime to P_K * q_mod
    for prime modulus q_mod (when chi = (./q_mod) equals +1 on [1, Y] and on all units/ideals).

    Here for each squarefree d_ell and odd prime r not dividing P_K * q_mod:
      - If r splits in F_ell ((Delta_ell / r) == 1), r O_{F_ell} = p_ell * p_ell'
        with phi_{F_ell, q}(p_ell) defined via unique root of t^2 == d_ell (mod r)...
      Actually for exact finite certification in Mode 2, we evaluate the exact multiplicative
      ideal-norm coefficients directly from the exact splitting Legendre symbol (Delta_ell / r)
      and the exact character value chi(r) = (r / q_mod) together with the F_2-linear splitting
      sign v_1(r) induced by the biquadratic compositum!
    """
    primes_K = first_k_primes(K)
    PK = 1
    for p in primes_K:
        PK *= p
    bad_prod = PK * q_mod

    # Check chi(a) == +1 for all 1 <= a <= Y coprime to q_mod
    chi_on_Y_all_plus = all(jacobi_symbol(a, q_mod) == 1 for a in range(1, Y + 1) if gcd(a, q_mod) == 1)

    keys = sorted(family.keys())
    # For each prime r <= X coprime to bad_prod, compute exact g(r) in F_2^K:
    # (-1)^{ell . g(r)} == (Delta_ell / r)
    # Since Delta_ell has same squarefree part as d_ell and r is odd & r nmid PK, (Delta_ell / r) = (d_ell / r)!
    # Under the cocycle-free extension (Lemma 5.1), for each r > Y with chi(r)=+1, v_1(r) is any F_2 vector
    # (taking v_1(r) = 0 corresponds to trivial splitting sign chi*(p_ell)=+1, or explicit F_2 vector from unit-character).
    # More importantly: compute exact b_hat_0(n) and b_0(n) + sum_{ell != 0} b_ell(n) == 2^K * b_hat_0(n) >= 0!
    b_sums_ell: Dict[str, int] = {k: 0 for k in keys}
    b0_total = 0
    smooth_count = 0
    coprime_count = 0
    all_wh_nonneg = True
    all_smooth_ge_1 = True

    for n in range(1, X + 1):
        if gcd(n, bad_prod) != 1:
            continue
        coprime_count += 1
        facs = prime_factorization_small(n)
        is_y_smooth = all(r <= Y for r, _ in facs)
        if is_y_smooth:
            smooth_count += 1

        # Compute exact b_ell(n) and b_0(n) using Lemma 5.1(2) closed form
        b0_n = 1
        b_ell_n: Dict[str, int] = {k: 1 for k in keys}
        b_hat_0_n = 1

        for r, e in facs:
            chi_r = jacobi_symbol(r, q_mod)
            # Determine splitting per ell: kr = (d_ell / r) in {-1, +1}
            if r <= Y or (chi_r == 1 and e % 2 == 0):
                b0_n *= (e + 1)
                b_hat_0_n *= ((e + 2) // 2)
                for k in keys:
                    leg = legendre_symbol(family[k]["d"], r)
                    if leg == 1:
                        b_ell_n[k] *= (e + 1)
                    else:
                        b_ell_n[k] *= (1 if e % 2 == 0 else 0)
            elif chi_r == 1 and e % 2 == 1:
                # Using v_1(r) = 0 (cocycle-free trivial sign on splitting primes)
                b0_n *= (e + 1)
                # b_hat_0 contribution = (e+1)/2 if v1(r)=0 else 0 (here >= 0 always!)
                b_hat_0_n *= ((e + 1) // 2)
                for k in keys:
                    leg = legendre_symbol(family[k]["d"], r)
                    b_ell_n[k] *= ((e + 1) if leg == 1 else 0)
            elif chi_r == -1 and e % 2 == 1:
                b0_n = 0
                b_hat_0_n = 0
                for k in keys:
                    b_ell_n[k] = 0
            else:  # chi_r == -1 and e == 2m even
                m_half = e // 2
                b0_n *= 1
                if m_half % 2 == 0:
                    b_hat_0_n *= 1
                else:
                    b_hat_0_n *= 0
                for k in keys:
                    leg = legendre_symbol(family[k]["d"], r)
                    if leg == 1:
                        b_ell_n[k] *= 1
                    else:
                        b_ell_n[k] *= (1 if m_half % 2 == 0 else -1)

        wh_sum = b0_n + sum(b_ell_n[k] for k in keys)
        if wh_sum < 0 or wh_sum % (1 << K) != 0:
            all_wh_nonneg = False
        if is_y_smooth and (wh_sum // (1 << K)) < 1:
            all_smooth_ge_1 = False

        b0_total += b0_n
        for k in keys:
            b_sums_ell[k] += b_ell_n[k]

    max_s_ell = max(b_sums_ell.values())
    total_s_ell = sum(b_sums_ell.values())
    # Exact Lemma 5.1(3) inequality: (2^K - 1) * max_ell S_ell(X) >= 2^K * Psi - b0_total
    lemma51_3_holds = ((1 << K) - 1) * max_s_ell >= (1 << K) * smooth_count - b0_total

    return {
        "chi_on_Y_all_plus": chi_on_Y_all_plus,
        "coprime_count": coprime_count,
        "smooth_count_Psi": smooth_count,
        "b0_total": b0_total,
        "max_S_ell": max_s_ell,
        "total_S_ell": total_s_ell,
        "all_wh_nonneg": all_wh_nonneg,
        "all_smooth_ge_1": all_smooth_ge_1,
        "lemma51_3_holds": lemma51_3_holds,
    }


# ---------------------------------------------------------------------------
# Section 7.4 & 8.1: Exact Bordered-Adjugate & Preconditioned Base-m Polynomials
# ---------------------------------------------------------------------------


def poly_eval_int(coeffs: Sequence[int], x: int) -> int:
    """Evaluate sum_j coeffs[j] * x^j (lowest degree first) by Horner's rule."""
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def poly_eval_mod(coeffs: Sequence[int], x: int, mod: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % mod
    return acc


def poly_deriv_int(coeffs: Sequence[int]) -> List[int]:
    return [j * coeffs[j] for j in range(1, len(coeffs))]


def poly_gcd_f2(a: List[int], b: List[int]) -> List[int]:
    """Exact polynomial GCD in F_2[X] (bits 0/1, lowest degree first)."""
    def trim(p: List[int]) -> List[int]:
        p = [c & 1 for c in p]
        while p and p[-1] == 0:
            p.pop()
        return p

    a, b = trim(a), trim(b)
    while b:
        # Polynomial remainder a mod b over F_2
        r = list(a)
        db = len(b) - 1
        while len(r) - 1 >= db:
            shift = len(r) - 1 - db
            for i in range(db + 1):
                r[shift + i] ^= b[i]
            r = trim(r)
        a, b = b, r
    return a


def is_irreducible_f2(coeffs: Sequence[int]) -> bool:
    """Exact Rabin irreducibility test for monic polynomial in F_2[X]."""
    p = [c & 1 for c in coeffs]
    while p and p[-1] == 0:
        p.pop()
    d = len(p) - 1
    if d < 1:
        return False

    def mul_mod(a: List[int], b: List[int]) -> List[int]:
        res = [0] * (len(a) + len(b))
        for i, ca in enumerate(a):
            if ca:
                for j, cb in enumerate(b):
                    if cb:
                        res[i + j] ^= 1
        while len(res) - 1 >= d:
            if res[-1]:
                shift = len(res) - 1 - d
                for i in range(d + 1):
                    res[shift + i] ^= p[i]
            res.pop()
        while res and res[-1] == 0:
            res.pop()
        return res

    # Compute X^(2^k) mod p for k = 1..d
    cur = [0, 1]  # X
    # Test gcd(p, X^(2^k) - X) == 1 for 1 <= k <= d // 2 (exact Lemma 8.1 proof!)
    for k in range(1, d + 1):
        cur = mul_mod(cur, cur)
        if k <= d // 2:
            diff = list(cur)
            if len(diff) < 2:
                diff.extend([0] * (2 - len(diff)))
            diff[1] ^= 1
            if poly_gcd_f2(p, diff) != [1]:
                return False
    # Also require X^(2^d) == X (mod p)
    return cur == [0, 1]


def exact_det_and_adjugate(mat: Sequence[Sequence[int]]) -> Tuple[int, List[List[int]]]:
    """Exact integer determinant and adjugate matrix via bordered-adjugate / Bareiss (zero floats)."""
    n = len(mat)
    if n == 0:
        return 1, []
    if n == 1:
        return mat[0][0], [[1]]

    def det_bareiss(A: List[List[int]]) -> int:
        k = len(A)
        if k == 0:
            return 1
        if k == 1:
            return A[0][0]
        M = [list(row) for row in A]
        sign = 1
        prev = 1
        for i in range(k - 1):
            if M[i][i] == 0:
                sw = -1
                for r in range(i + 1, k):
                    if M[r][i] != 0:
                        sw = r
                        break
                if sw == -1:
                    return 0
                M[i], M[sw] = M[sw], M[i]
                sign = -sign
            piv = M[i][i]
            for r in range(i + 1, k):
                for c in range(i + 1, k):
                    M[r][c] = (M[r][c] * piv - M[r][i] * M[i][c]) // prev
            prev = piv
        return sign * M[k - 1][k - 1]

    d = det_bareiss([list(r) for r in mat])
    adj: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [
                [mat[r][c] for c in range(n) if c != j]
                for r in range(n)
                if r != i
            ]
            cof = det_bareiss(minor)
            adj[j][i] = cof if ((i + j) % 2 == 0) else -cof
    return d, adj


def lcm_1_to_k(k_idx: int) -> int:
    """Exact L_idx = prod_{p' <= k_idx} (p')^{e_{p'}} = lcm(1, 2, ..., k_idx) (Lemma 11.1)."""
    acc = 1
    for t in range(2, k_idx + 1):
        acc = lcm_int(acc, t)
    return acc
