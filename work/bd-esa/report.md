# bd-esa

**Problem:** Essential self-adjointness of chi(xi) + chi'(eta) for the Bullough-Dodd S-matrix

- S: Known for every n: chi_n(xi) = n P_n A_1^2 P_n and chi'_n(eta) = n P_n B_n^2 P_n are each positive self-adjoint on H_n intersected with Dom(A_1^2), resp. Dom(B_n^2) (Theorem A_n), and AB is bounded (Tanimoto)
- G: Proven: the sum chi_n(xi) + chi'_n(eta) is self-adjoint on the intersection of their domains for every n, so (HA-T) holds for the Bullough-Dodd S-matrix at B = 1/2

20 nodes, 182 edges scored, 20 usable (p >= 0.5); judge: Claude (Laya shortlists)

## Best paths (product of edge probabilities)

1. **0.2622**  S -> F3 -> F9 -> B7 -> B2 -> B6 -> B1 -> G
2. **0.2428**  S -> F3 -> F9 -> B7 -> B2 -> B6 -> G
3. **0.2247**  S -> F3 -> F8 -> F9 -> B7 -> B2 -> B6 -> B1 -> G
4. **0.2081**  S -> F3 -> F8 -> F9 -> B7 -> B2 -> B6 -> G
5. **0.1966**  S -> F3 -> F9 -> B7 -> B2 -> B6 -> B1 -> B3 -> G

## Winning path, step by step

- `S` Known for every n: chi_n(xi) = n P_n A_1^2 P_n and chi'_n(eta) = n P_n B_n^2 P_n are each positive self-adjoint on H_n intersected with Dom(A_1^2), resp. Dom(B_n^2) (Theorem A_n), and AB is bounded (Tanimoto)
  - p=0.850
- `F3` Truncate one side: replace B_n^2 by the bounded operator min(B_n^2, M), so Z_A + Z_{B,M} is self-adjoint by bounded perturbation
  - p=0.700
- `F9` Bound the same-slot cross term by c ||A_1 u|| ||min(B_n, sqrt M) u|| using that AB is bounded
  - p=0.800
  - gap: combine with the non-negative coset terms and Young's inequality
- `B7` 2 Re <Z_A u, Z_{B,M} u> is bounded below by -eps ||Z_A u||^2 - C ||u||^2, uniformly in M
  - p=0.800
  - gap: needs F3 for the existence of u_M
- `B2` For every f in H_n there are u_M with (Z_A + Z_{B,M} + 1) u_M = f and bounds on ||Z_A u_M|| and ||Z_{B,M} u_M|| uniform in M
  - p=0.850 **HARD** (big jump, cos 0.49)
  - gap: closedness of Z_A, Z_B and weak compactness; Z_{B,M} u_M -> Z_B u needs the truncation limit
- `B6` Pass to weak limits as M goes to infinity: u_M converges to u in the domain with (T + 1) u = f
  - p=0.900 **HARD** (big jump, cos 0.32)
- `B1` The sum T is a positive symmetric operator with Ran(T + 1) equal to all of H_n
  - p=0.900 **HARD** (big jump, cos 0.45)
  - gap: apply the range criterion (B5) and take the direct sum over n
- `G` Proven: the sum chi_n(xi) + chi'_n(eta) is self-adjoint on the intersection of their domains for every n, so (HA-T) holds for the Bullough-Dodd S-matrix at B = 1/2

## Strongest alternative waypoints (best path through node / best path)

- 1.00  `F3` Truncate one side: replace B_n^2 by the bounded operator min(B_n^2, M), so Z_A + Z_{B,M} is self-adjoint by bounded perturbation
- 1.00  `F9` Bound the same-slot cross term by c ||A_1 u|| ||min(B_n, sqrt M) u|| using that AB is bounded
- 1.00  `B1` The sum T is a positive symmetric operator with Ran(T + 1) equal to all of H_n
- 1.00  `B2` For every f in H_n there are u_M with (Z_A + Z_{B,M} + 1) u_M = f and bounds on ||Z_A u_M|| and ||Z_{B,M} u_M|| uniform in M
- 1.00  `B6` Pass to weak limits as M goes to infinity: u_M converges to u in the domain with (T + 1) u = f
- 1.00  `B7` 2 Re <Z_A u, Z_{B,M} u> is bounded below by -eps ||Z_A u||^2 - C ||u||^2, uniformly in M
- 0.86  `F8` Expand the cross term <Z_A u, Z_{B,M} u> over cosets of S_n: B acting on slots 2 to n commutes with A_1^2, giving a non-negative contribution
- 0.75  `B3` The adjoint equation T* v = -v has only the solution v = 0

## Hard links to decode

- `bdnm.py bridge bd-esa B2 B6`
- `bdnm.py bridge bd-esa B6 B1`
- `bdnm.py bridge bd-esa B1 G`