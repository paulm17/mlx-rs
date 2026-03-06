## Exact MoE Router Design for Rust MLX

### Goal

Close the remaining MoE gap against Python without destabilizing the dense-model wins already achieved.

Current stable benchmark position:

- `Llama`: near parity
- `Qwen3`: at or above parity
- `Qwen1.5-MoE`: about 50% of Python decode throughput
- `LFM2`: about 60% of Python decode throughput

The remaining loss is now concentrated in MoE routing, especially single-token decode routing.

### What the current profiles prove

For the stable Rust MoE paths:

- `Qwen1.5-MoE` decode is dominated by `moe_router_host_s`
- `LFM2` decode is also router-host heavy after the selected-expert batching win
- expert execution is no longer the main bottleneck
- dense decode-loop synchronization is no longer the main bottleneck

That means the next real win is not another generic runtime tweak. It is a router redesign.

### Design constraints

The previous direct device-router attempts failed for one of two reasons:

- they changed routing semantics enough to alter stopping behavior
- they reduced transfer volume but introduced more synchronization cost than they saved

So the design must be staged:

1. prove exact semantic equivalence first
2. measure where the cost moves
3. only then switch live routing

### Target router behavior

For single-token decode:

1. compute router logits on device
2. compute router probabilities on device
3. compute top-k expert indices on device
4. gather selected top-k probabilities on device
5. copy only selected expert ids and selected probabilities to host
6. preserve deterministic ordering on host:
   - probability descending
   - expert id ascending for ties
7. feed the unchanged selected experts and weights into the existing expert execution path

For multi-token prefill:

- keep the current stable host router path until single-token device routing is proven correct and beneficial

This keeps the largest decode hotspot targeted first while avoiding a broad prefill rewrite.

### Phase plan

#### Phase 1: Shadow Validation

Do not change live routing.

For single-token MoE decode only:

- compute the current stable host router result
- compute the experimental device router result in shadow mode
- compare:
  - selected expert ids
  - selected probabilities
- record mismatch counters in the generation profile

Activation:

- `MLX_VALIDATE_MOE_DEVICE_ROUTER=1`
- `MLX_VALIDATE_MOE_DEVICE_ROUTER=qwen`
- `MLX_VALIDATE_MOE_DEVICE_ROUTER=lfm2`
- `MLX_VALIDATE_MOE_DEVICE_ROUTER=all`

Success criteria:

- zero expert-id mismatches across repeated benchmark runs
- probability differences stay within a small floating-point tolerance
- no change in output text or stop behavior

#### Phase 2: Opt-in Live Switch

After shadow validation is clean:

- add a separate env flag for live routing switch
- keep the host router as default
- enable device router only for validated models

Activation:

- `MLX_EXPERIMENTAL_MOE_DEVICE_ROUTER=qwen`
- `MLX_EXPERIMENTAL_MOE_DEVICE_ROUTER=lfm2`

Success criteria:

- same stop behavior as baseline
- same token outputs as baseline for deterministic decode
- router-host time drops materially
- total decode throughput improves

#### Phase 3: Default Adoption

Only switch the default path when all of these hold:

- stable across multiple benchmark runs
- no stop-token regressions
- no repeated-text regressions
- sustained throughput gain, not one-off noise

### Exact implementation shape

#### Shared instrumentation

Extend `MoeProfileStats` with:

- `device_router_shadow_checks`
- `device_router_shadow_mismatches`

Propagate these into:

- `mlx-models` aggregate profile
- `GenerationProfile`
- server `debug.profile`

#### Qwen1.5-MoE phase-1 path

In the single-token fast path:

1. keep the existing host path as the source of truth
2. when validation is enabled:
   - run `softmax(-1)` on the device router logits
   - run `argpartition` on probabilities on device
   - slice the last `k` entries from the partition result
   - gather selected top-k probabilities with `take_along_axis`
   - copy only those selected ids and probabilities to host
   - sort host-side for deterministic comparison
3. compare against the host router result
4. record mismatch counters

Do not use the device-selected result to drive expert execution in phase 1.

#### LFM2 phase-1 path

Use the same shadow-validation structure in the single-token fast path.

The live path remains:

- host logits copy
- CPU top-k / weight reconstruction
- selected-expert batched execution

#### Comparison policy

For each selected slot:

- expert id must match exactly
- probability difference tolerance: `1e-4`

Any mismatch increments the mismatch counter.

### Why this is the correct next step

The unstable part of the earlier experiments was not the idea of device routing itself. It was switching behavior too early.

This design fixes that:

- the benchmarked path stays stable
- correctness becomes measurable
- the next live switch can be justified by actual mismatch data

### What should happen next after phase 1 lands

1. run baseline with validation disabled
2. run `MLX_VALIDATE_MOE_DEVICE_ROUTER=qwen`
3. run `MLX_VALIDATE_MOE_DEVICE_ROUTER=lfm2`
4. inspect:
   - mismatch counters
   - router-host time
   - any behavioral drift
5. only then add a live device-router switch for the validated model
