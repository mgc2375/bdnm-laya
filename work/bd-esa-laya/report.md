# bd-esa-laya

**Problem:** Essential self-adjointness of chi(xi) + chi'(eta) for the Bullough-Dodd S-matrix

- S: Known for every n: chi_n(xi) = n P_n A_1^2 P_n and chi'_n(eta) = n P_n B_n^2 P_n are each positive self-adjoint on H_n intersected with Dom(A_1^2), resp. Dom(B_n^2) (Theorem A_n), and AB is bounded (Tanimoto)
- G: Proven: the sum chi_n(xi) + chi'_n(eta) is self-adjoint on the intersection of their domains for every n, so (HA-T) holds for the Bullough-Dodd S-matrix at B = 1/2

20 nodes, 91 edges scored, 85 usable (p >= 0.341); judge: tuned Laya: raw scores, forward-consistent edges, top half usable

## Best paths (product of edge probabilities)

1. **0.2612**  S -> F3 -> G
2. **0.0824**  S -> F9 -> B7 -> F3 -> G
3. **0.0155**  S -> F4 -> F10 -> F8 -> F9 -> B7 -> F3 -> G

## Winning path, step by step

- `S` Known for every n: chi_n(xi) = n P_n A_1^2 P_n and chi'_n(eta) = n P_n B_n^2 P_n are each positive self-adjoint on H_n intersected with Dom(A_1^2), resp. Dom(B_n^2) (Theorem A_n), and AB is bounded (Tanimoto)
  - p=0.548
- `F3` Truncate one side: replace B_n^2 by the bounded operator min(B_n^2, M), so Z_A + Z_{B,M} is self-adjoint by bounded perturbation
  - p=0.477 **HARD** (low p)
- `G` Proven: the sum chi_n(xi) + chi'_n(eta) is self-adjoint on the intersection of their domains for every n, so (HA-T) holds for the Bullough-Dodd S-matrix at B = 1/2

## Strongest alternative waypoints (best path through node / best path)

- 1.00  `F3` Truncate one side: replace B_n^2 by the bounded operator min(B_n^2, M), so Z_A + Z_{B,M} is self-adjoint by bounded perturbation
- 0.94  `B4` The form sum of chi_n(xi) and chi'_n(eta) has operator domain equal to the intersection of their domains
- 0.32  `F9` Bound the same-slot cross term by c ||A_1 u|| ||min(B_n, sqrt M) u|| using that AB is bounded
- 0.32  `B7` 2 Re <Z_A u, Z_{B,M} u> is bounded below by -eps ||Z_A u||^2 - C ||u||^2, uniformly in M
- 0.27  `F4` Check whether chi_n(xi) and chi'_n(eta) strongly commute, so their sum is self-adjoint by joint spectral calculus
- 0.23  `F2` Take the KLMN form sum of the two positive operators chi_n(xi) and chi'_n(eta)
- 0.22  `B2` For every f in H_n there are u_M with (Z_A + Z_{B,M} + 1) u_M = f and bounds on ||Z_A u_M|| and ||Z_{B,M} u_M|| uniform in M
- 0.22  `B6` Pass to weak limits as M goes to infinity: u_M converges to u in the domain with (T + 1) u = f

## Hard links to decode

- `bdnm.py bridge bd-esa-laya F3 G`