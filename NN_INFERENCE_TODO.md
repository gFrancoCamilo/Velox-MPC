# NN inference on Velox — deferred work

Branch `nn_inference` replaces the anonymous-broadcast mixing circuit with dense
neural-network inference. Preprocessing, input, and online phases are implemented
and verified; the items below are known gaps.

## 1. Rolling / streaming tuple verification  (blocking for malicious security)

`Context::verification_enabled` is `false`, so `verf_state` records nothing and
the online phase goes straight to output reconstruction. **A malicious party can
corrupt the inference result undetected; every current benchmark number is
semi-honest-only.**

Verification cannot simply be switched back on. It delinearizes over *component*
products, of which there are `b(2x² + xy)` — 9.6e9 at `b=256, x=4096`, i.e. ~154 GB
of retained `x`/`y` shares at 8 bytes each, before the compression levels allocate
anything.

The tuples are hugely redundant, which is the way out: the `y` side is only the
weight matrix (`2x² + xy` distinct elements, reused across every example and every
column position) and the `x` side only the activations (`3bx`). Store verification
state **symbolically** as `(activation slice, weight column, r^g)` over that compact
base and stream each compression level. The coin for a level arrives only *after*
that level's multiplication completes, so level 0 needs a second streaming pass
rather than a cached `x_polys`.

## 2. Generalize `verf_state` to inner products

`VerificationState::add_mult_inputs` (`tuple_verification/verf_state.rs`) stores only
`x[0]` / `y[0]` per gate. That was lossless for the mixing circuit's length-1 gates,
but for an inner product it drops `d_in - 1` of `d_in` components, making the check
vacuous. Needs per-gate `Vec<Vec<LargeField>>` with `r^g` applied to every component
of gate `g` and to `z_g`. `init_compression_level` itself needs no change — it just
receives a longer vector.

## 3. Field soundness after verification returns

`LargeField` is now the base Mersenne-61 prime field (8 bytes), so a single
field-element check has ~2^-61 soundness. Irrelevant while verification is off; a
restored verification phase must repeat its checks or run them over an extension.

## 4. Memory: `LargeFieldSer = Vec<u8>`

Every serialized field element is an individually heap-allocated `Vec<u8>` holding
8 bytes — roughly 48 bytes of `Vec` header plus allocator overhead per 8 bytes of
payload. The ratio was ~1.5x when elements were 32-byte Fp4; at 8 bytes it is ~6x,
and it applies to every ACSS payload and protocol message. Changing the alias to
`[u8; 8]` removes the per-element allocation entirely, but touches `acss_ab`,
`sh2t`, `avid_ab`, and `mpc`. Measured context: a `n=4, x=2048, y=512, b=8` run
(9.4M weights = 75 MB of assembled shares) peaked at 1111 MB RSS per party.

## 5. Liveness at the input barrier

The input phase waits for **all n** dealers, by design, so the assembled weight
matrix has no holes. One crashed party stalls the run. If that bites, the
alternatives are an ACS core set with a public default for missing blocks, or
`t+1`-fold block replication.

## 6. Output-mask traffic

Public mask reconstruction broadcasts every mask share from every origin:
`b·y × 8 B × n × n`. At `b=256, y=1000, n=16` that is ~520 MB.

## 7. Re-check the zeroed-`[o]` fix once verification returns

`gen_2t_sharings` filled only 1/3 of its groups; the rest folded to *exactly zero*
through the Vandermonde multiply, yielding degree-2t "masks" of zero that mask
nothing in the DN reduction. Fixed in `rand_sh.rs`, but it is a masking leak rather
than a correctness bug, so no test would have caught it — re-derive the argument
when the security proof matters again.
