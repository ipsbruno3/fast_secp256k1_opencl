# High-Performance secp256k1 OpenCL Kernels with BIP32 Derivation

This repository provides optimized OpenCL kernels for elliptic curve operations on the secp256k1 curve, commonly used in cryptocurrencies like Bitcoin. The implementation is tailored for GPU architectures (AMD/NVIDIA), emphasizing low-latency field arithmetic, Jacobian point operations, wide-window Comb scalar multiplication, batch affine conversion via Blelloch scan, and integrated BIP32 key derivation using optimized SHA512 and HMAC-SHA512.

## Key Features and Optimizations

### Field Arithmetic (`math.cl`)
Operations over GF(p) where `p = 2^256 - 2^32 - 977`, using 4×64-bit little-endian limbs for alignment with GPU registers.

- **Fast Modular Reduction (`fe_reduce512`)**: Exploits `p`'s special form for efficient 512-bit product reduction. Multiplies high 256 bits by `977 + 2^32` using `mul_hi` and shifts; propagates carries with up to 3 iterations; final 2× conditional subtract `p`. Constant-time and branchless where possible.

- **Multiplication (`fe_mul`, `mul256_raw`)**: Schoolbook algorithm with manual 128-bit accumulation via `acc_add128`. Uses `mul_hi` for high bits of 64×64 products; macro-based partial product accumulation reduces register pressure. Full schoolbook 4×64 multiplication producing 8 limbs (512-bit). Accumulation uses explicit 128-bit intermediates (via `mul_hi` + low product + carry). Reduction is applied immediately after to keep values bounded.

- **Squaring (`fe_sqr`, `sqr256_raw`)**: Specialized version saving ~25-30% multiplications by doubling cross terms (`MAC2` macro with shift-carry handling). Faster than generic multiplication. Squaring reuses symmetry: cross terms are doubled, diagonal terms computed once. This saves multiplications vs a generic multiply path. In practice, squaring is expected to be noticeably cheaper than general multiplication in the field.

- **Addition/Subtraction (`fe_add`, `fe_sub`)**: Fully branchless using `bitselect` masks from carry/borrow detection. Conditional modular correction integrated. Compute raw sum/diff with explicit carry/borrow. Correct overflow/underflow by conditionally adding/subtracting `p`. Use bit masks / bitselect-style operations instead of branching where possible. Reason: branches are expensive on GPUs when lanes diverge.

- **Inversion (`fe_inv`)**: Fermat's Little Theorem (`a^{p-2}`) with custom addition chain: builds powers like `x88 = a^{2^88-1}`; assembles `p-3` with minimal multiplies (~250 squarings + few muls). Unrolled repeated squarings via macro. The exponentiation chain is built with an optimized sequence of squarings and multiplies: build small “(2^k − 1)” blocks (e.g., x11, x22, x44, x88…); compose them to reach the full exponent with a minimal count of multiplies. Keep the chain tight and predictable (ideal for unrolling / scheduling). Inversion is still expensive compared to mul/sqr, which is why batch affine conversion is critical.

- **Scalar Ops Modulo n**: Big-endian add/sub/compare for curve order `n`; conditional reduction in `addmod_n`.

### Point Operations (`point.cl`)
Uses Jacobian coordinates (`x, y, z`) for inversion-free additions during scalar mul; infinity at `z=0`. Point representation and formulas: points are processed in Jacobian form to avoid inversions during point addition/doubling. Point at infinity is typically represented by `Z = 0`.

- **Doubling (`point_double64`)**: Unrolled Jacobian formula (12M + 5S); early exits for infinity or `y=0`.

- **Full Addition (`point_add_64`)**: Standard Jacobian add (16M + 7S); handles equal points via doubling.

- **Mixed Addition (`point_add_affine`)**: Adds Jacobian + affine (Z=1) point (11M + 5S); key for Comb efficiency. When one point is affine (Z=1), mixed addition saves field multiplications and is a major speed win in precomputation-based scalar multiplication.

- **Affine Conversion (`point_to_affine`)**: Single-point inversion with `z^{-1}`, then `x' = x * z^{-2}`, `y' = y * z^{-3}`.

- **Batch Affine Conversion (`point_to_affine_batch64`)**: Parallel Montgomery inversion for 64 points using Blelloch scan in local memory. Computes exclusive prefix products; inverts total product once; derives individual inverses via prefix/suffix. O(log N) time, ~40x speedup over serial inversions. Masks infinity points. Affine conversion requires `inv(Z)`: `x_aff = X / Z²`, `y_aff = Y / Z³`. Instead of inverting each `Z` independently, the code uses a standard batch trick: compute prefix products of all Z values in local memory (parallel scan); invert the total product once; recover each inv(Zᵢ) using prefix/suffix products and the inverted total. Implementation detail: parallel prefix product (Blelloch scan)—uses a workgroup-local scan to build exclusive prefix products; requires a fixed workgroup size (commonly 64). Greatly reduces total inversion cost: 1 inversion + O(N) multiplications instead of N inversions. This is one of the highest-impact throughput optimizations for elliptic-curve pipelines.

### Scalar Multiplication
This project uses a **Comb** method for `k * G` (base point multiplication), trading memory for speed. Typical Comb parameters: `w` = window width (bits), `d` = number of columns. A common high-throughput configuration is a wide window that creates a table of size `2^w` points per column (or a combined layout), yielding a table on the order of a few MB. The exact table layout is generated by `comb.py` and consumed by the OpenCL kernels.

- **Comb Method (`point_mulG_comb8`, w=16)**: Fixed-base (k * G) with 2^16 precomputed affine points (~4MB table). 16 iterations: 1 doubling + conditional mixed add per column. Nibble indexing via bit shifts/masks. Optimal for fixed-base; fewer ops/bit than WNAF when precompute is feasible. High-level algorithm: for each column (from most significant to least)—perform a doubling step (or a structured set of doublings depending on comb variant); extract a `w`-bit index from the scalar; conditionally mixed-add the table point at that index. This is generally faster than wNAF for fixed-base multiplication when a sufficiently large table is feasible.

### SHA512 Optimizations
Integrated for HMAC in BIP32; fully unrolled 80-round compression. The SHA-512 implementation is structured for GPU efficiency: unrolled rounds where it matters; rolling schedule windows to avoid storing full W[0..79]; bitwise ops mapped to OpenCL intrinsics where beneficial (`rotate`, `bitselect`). 64-bit state in registers (A0..A7) and constants inlined (K literals). Uses OpenCL intrinsics: rotate() for ROR and bitselect() for Ch/Maj (branchless). Message schedule computed on-the-fly with a rolling window (W16..W32) to avoid W[80] and reduce private memory/register pressure. Pure compression: expects `message` already as 16xulong words per block (formatting done by caller). sha512_hash_two_blocks_message() compresses exactly 2 blocks; no padding performed here.

- **Unrolled Rounds**: All 80 rounds explicit via `RoR` macro; eliminates loop overhead and divergence.

- **Sigma Functions (`L0`, `L1`, `SHA512_S0`, `SHA512_S1`)**: Use `rotate` intrinsic for rotations; XOR/shifts for constant-time.

- **Message Schedule Optimization**: Partial scheduling with discrete W variables (W16-W32); computes only necessary W in later blocks (e.g., 9+5 in final).

- **Ch/Maj Functions (`F1`, `F0`)**: Branchless via `bitselect`.

- **Two-Block Hash (`sha512_hash_two_blocks_message`)**: Processes two 1024-bit blocks sequentially; tailored for HMAC's fixed sizes.

### HMAC-SHA512 Optimizations (`hmac512_ccode_msg37`)
Specialized for BIP32 CKD messages (37 bytes). Implements standard HMAC: inner = SHA512((K ⊕ ipad) || msg); outer = SHA512((K ⊕ opad) || inner_digest). This is the foundation for BIP32/BIP84-style derivations (IL/IR splits).

- **Fixed-Size Specialization**: Hardcodes 37-byte message; direct block setup for inner/outer (no loops for padding).

- **Precomputed Pads**: `IPAD/OPAD` as constants; unrolled XOR to key.

- **Direct Integration**: Sets inner block with message + 0x80 + zeros + len(1320); hashes to temp; sets outer with temp + pad + len(1536).

- **No General Loops**: All assignments explicit; reduces instructions for crypto paths.

### BIP32/BIP84-Style Derivation Glue (`derive.cl`)
The derivation logic typically follows: hardened derivation uses parent private key material (HMAC input includes 0x00 || k || index); normal derivation uses parent public key material (HMAC input includes serP(point) || index). For BIP84 paths like `m/84'/0'/0'/change/index`: first levels are hardened (no public key required); later levels are normal (public key required); public keys are produced via `k*G` and then used in subsequent derivation steps. The kernels here provide the low-level building blocks; you can integrate them into a full pipeline (address derivation, bloom/tag checks, batching, etc.). Total de multiplicações escalares: apenas 3 (account, change, receive/address index). Todas as pubkeys são calculadas via batch de 64 (mesmo se for só 1 por workgroup → ainda vale pela inversão única).

## Performance Highlights
- **Memory**: Comb table ~4MB global; ~6KB local per WG.
- **Constant-Time**: Critical paths (field ops, SHA) avoid timing leaks.

## Usage Guide for secp256k1 OpenCL Kernels
The implementation focuses on high-performance elliptic curve operations on GPUs, including field arithmetic, point operations, and BIP-32/BIP-84 key derivation using the Comb method for scalar multiplication (no WNAF).

**Key Requirements**:
- **OpenCL Setup**: Use PyOpenCL for host-side management. Ensure your GPU supports OpenCL 1.2+ with 64-bit integers.
- **Precompute Comb Table**: The Comb method requires a precomputed table of 2^16 affine points (for W=16). Compute this once on the host and cache it (as Numpy array). Allocate the buffer only once for efficiency.
- **Workgroup Size**: Fixed at 64 (WG=64) for batch inversion via Blelloch scan.
- **Memory**: Comb table is ~4MB (65536 entries × 64 bytes). Use `cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR` for host-pinned memory (no perf loss) or `cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR` for VRAM copy.
- **Inputs/Outputs**: Scalars and points are 4x `ulong` (little-endian for points, big-endian for scalars in some ops).

### Benchmarks
| Operation                          | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Std Dev (ms) | Throughput (ops/sec) |
|------------------------------------|---------------|---------------|---------------|--------------|-----------------------|
| 1) Comb Mul + Batch Affine (point_to_affine_batch64) | 108.23       | 107.49       | 109.28       | 0.57        | 92,843,625           |
| 2) Comb Mul Only (Jacobian Output) | 58.82        | 56.99        | 59.30        | 0.68        | 170,824,684          |
| 3) Comb Mul + Individual Affine (xy64_to_affine) | 105.57       | 105.04       | 106.01       | 0.26        | 95,188,408           |
| 4) Point Add Only (point_add_64, Jacobian) | 169.53       | 169.49       | 169.61       | 0.04        | 3,793,515,429        |
| 5) Point Add + Batch Affine (point_add_64 + point_to_affine_batch64) | 224.74       | 224.11       | 225.29       | 0.32        | 2,861,534,456        |
| 6) Point Add + Individual Affine (point_add_64 + point_to_affine) | 230.76       | 230.15       | 231.04       | 0.26        | 2,786,943,993        |

### Device Info
- Device: NVIDIA GeForce RTX 5090
- Config: NUM_ITEMS=10048576; WG=64; Groups=157009; Runs=10 (1 warm-up); ADD_ITERS=64

**Benchmark Code**

```python
import pyopencl as cl
import numpy as np
import os, time
import comb

# -------------------------
# Config
# -------------------------
WG_SIZE   = 64
NUM_ITEMS = 10_048_576
NUM_ITEMS -= (NUM_ITEMS % WG_SIZE)
NUM_RUNS  = 10     # 1 warm-up + 5 medidos
ADD_ITERS = 64     # aumente (ex: 32/128) se o tempo ficar pequeno demais

CL_DIR='.'
MATH_CL   = os.path.join(CL_DIR,'math.cl')
POINT_CL  = os.path.join(CL_DIR,'point.cl')
COMMON_CL = os.path.join(CL_DIR,'common.cl')

# -------------------------
# OpenCL setup
# -------------------------
platforms = cl.get_platforms()
devs = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=[devs[0]])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags

with open(MATH_CL,'r',encoding='utf-8') as f: math_src=f.read()
with open(POINT_CL,'r',encoding='utf-8') as f: point_src=f.read()
with open(COMMON_CL,'r',encoding='utf-8') as f: common_src=f.read()

# -------------------------
# Extra kernels (wrappers)
# -------------------------
bench_kernels = r"""


// 1) mul + affine batch
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_mul_batch_affine(
    __global const ulong *k_be,
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB,
    __local ulong *lz,
    __local ulong *lprefix,
    __local ulong *lscan,
    __local ulong *linvTot
){
    const uint gid = get_global_id(0);
    ulong kb[4] = { k_be[gid*4+0], k_be[gid*4+1], k_be[gid*4+2], k_be[gid*4+3] };

    ulong k_le[4];
    scalar_be64_to_le64_4(kb, k_le);

    jac_t R;
    point_mulG_comb8(&R, k_le, COMB);

    const int inf = point_is_inf(&R);
    ulong ax[4], ay[4];
    point_to_affine_batch64(ax, ay, &R, inf, lz, lprefix, lscan, linvTot);

    const uint o = gid*4;
    X_out[o+0]=ax[0]; X_out[o+1]=ax[1]; X_out[o+2]=ax[2]; X_out[o+3]=ax[3];
    Y_out[o+0]=ay[0]; Y_out[o+1]=ay[1]; Y_out[o+2]=ay[2]; Y_out[o+3]=ay[3];
}

// 2) mul only (writes Jacobian X/Y)
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_mul_only(
    __global const ulong *k_be,
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB
){
    const uint gid = get_global_id(0);
    ulong kb[4] = { k_be[gid*4+0], k_be[gid*4+1], k_be[gid*4+2], k_be[gid*4+3] };

    ulong k_le[4];
    scalar_be64_to_le64_4(kb, k_le);

    jac_t R;
    point_mulG_comb8(&R, k_le, COMB);

    const uint o = gid*4;
    X_out[o+0]=R.x[0]; X_out[o+1]=R.x[1]; X_out[o+2]=R.x[2]; X_out[o+3]=R.x[3];
    Y_out[o+0]=R.y[0]; Y_out[o+1]=R.y[1]; Y_out[o+2]=R.y[2]; Y_out[o+3]=R.y[3];
}

// 3) xy64_to_affine (affine individual)
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_xy64_to_affine(
    __global const ulong *k_be,
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB
){
    const uint gid = get_global_id(0);
    ulong kb[4] = { k_be[gid*4+0], k_be[gid*4+1], k_be[gid*4+2], k_be[gid*4+3] };

    ulong ax[4], ay[4];
    xy64_to_affine(ax, ay, kb, COMB);

    const uint o = gid*4;
    X_out[o+0]=ax[0]; X_out[o+1]=ax[1]; X_out[o+2]=ax[2]; X_out[o+3]=ax[3];
    Y_out[o+0]=ay[0]; Y_out[o+1]=ay[1]; Y_out[o+2]=ay[2]; Y_out[o+3]=ay[3];
}

// Helpers for add benches: pick 2 affine points from COMB and make Jacobian (z=1)
static inline void load_affine_from_comb(__private ulong Qx[4], __private ulong Qy[4],
                                        __global const ulong *COMB, uint idx){
    const __global ulong *p = COMB + ((size_t)idx << 3);
    Qx[0]=p[0]; Qx[1]=p[1]; Qx[2]=p[2]; Qx[3]=p[3];
    Qy[0]=p[4]; Qy[1]=p[5]; Qy[2]=p[6]; Qy[3]=p[7];
}

// 4) point_add only (Jacobian), does iters adds: A = A + B
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_add_only(
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB,
    const uint iters
){
    const uint gid = get_global_id(0);
    const uint mask = (COMB_T - 2u);
    const uint idx1 = (gid & mask) + 1u;
    const uint idx2 = ((gid * 0x9e3779b9u + 7u) & mask) + 1u;

    ulong Ax[4], Ay[4], Bx[4], By[4];
    load_affine_from_comb(Ax, Ay, COMB, idx1);
    load_affine_from_comb(Bx, By, COMB, idx2);

    jac_t A, B, R;
    point_set_affine(&A, Ax, Ay);
    point_set_affine(&B, Bx, By);

    for(uint i=0;i<iters;i++){
        point_add_64(&R, &A, &B);
        point_copy(&A, &R);
    }

    const uint o = gid*4;
    X_out[o+0]=A.x[0]; X_out[o+1]=A.x[1]; X_out[o+2]=A.x[2]; X_out[o+3]=A.x[3];
    Y_out[o+0]=A.y[0]; Y_out[o+1]=A.y[1]; Y_out[o+2]=A.y[2]; Y_out[o+3]=A.y[3];
}

// 5) point_add + affine batch
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_add_batch_affine(
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB,
    const uint iters,
    __local ulong *lz,
    __local ulong *lprefix,
    __local ulong *lscan,
    __local ulong *linvTot
){
    const uint gid = get_global_id(0);
    const uint mask = (COMB_T - 2u);
    const uint idx1 = (gid & mask) + 1u;
    const uint idx2 = ((gid * 0x9e3779b9u + 7u) & mask) + 1u;

    ulong Ax[4], Ay[4], Bx[4], By[4];
    load_affine_from_comb(Ax, Ay, COMB, idx1);
    load_affine_from_comb(Bx, By, COMB, idx2);

    jac_t A, B, R;
    point_set_affine(&A, Ax, Ay);
    point_set_affine(&B, Bx, By);

    for(uint i=0;i<iters;i++){
        point_add_64(&R, &A, &B);
        point_copy(&A, &R);
    }

    const int inf = point_is_inf(&A);
    ulong ax[4], ay[4];
    point_to_affine_batch64(ax, ay, &A, inf, lz, lprefix, lscan, linvTot);

    const uint o = gid*4;
    X_out[o+0]=ax[0]; X_out[o+1]=ax[1]; X_out[o+2]=ax[2]; X_out[o+3]=ax[3];
    Y_out[o+0]=ay[0]; Y_out[o+1]=ay[1]; Y_out[o+2]=ay[2]; Y_out[o+3]=ay[3];
}

// 6) point_add + affine individual
__kernel __attribute__((reqd_work_group_size(WG,1,1)))
void k_add_indiv_affine(
    __global ulong *X_out,
    __global ulong *Y_out,
    __global const ulong *COMB,
    const uint iters
){
    const uint gid = get_global_id(0);
    const uint mask = (COMB_T - 2u);
    const uint idx1 = (gid & mask) + 1u;
    const uint idx2 = ((gid * 0x9e3779b9u + 7u) & mask) + 1u;

    ulong Ax[4], Ay[4], Bx[4], By[4];
    load_affine_from_comb(Ax, Ay, COMB, idx1);
    load_affine_from_comb(Bx, By, COMB, idx2);

    jac_t A, B, R;
    point_set_affine(&A, Ax, Ay);
    point_set_affine(&B, Bx, By);

    for(uint i=0;i<iters;i++){
        point_add_64(&R, &A, &B);
        point_copy(&A, &R);
    }

    ulong ax[4], ay[4];
    point_to_affine(ax, ay, &A);

    const uint o = gid*4;
    X_out[o+0]=ax[0]; X_out[o+1]=ax[1]; X_out[o+2]=ax[2]; X_out[o+3]=ax[3];
    Y_out[o+0]=ay[0]; Y_out[o+1]=ay[1]; Y_out[o+2]=ay[2]; Y_out[o+3]=ay[3];
}
"""

full_src = common_src + math_src + point_src + bench_kernels
prg = cl.Program(ctx, full_src).build(options='-I .')

# -------------------------
# Buffers
# -------------------------
table_comb = comb.build_comb_table_u64()
buf_comb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=table_comb)

# random scalars (4x u64 por item)
rng = np.random.default_rng(12345)
k_words = rng.integers(0, np.iinfo(np.uint64).max, size=(NUM_ITEMS,4), dtype=np.uint64)
zmask = np.all(k_words == 0, axis=1)
k_words[zmask,3] = 1
k_be_np = np.ascontiguousarray(k_words.reshape(-1))
buf_k_be = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_be_np)

x_out_np = np.empty(NUM_ITEMS*4, dtype=np.uint64)
y_out_np = np.empty(NUM_ITEMS*4, dtype=np.uint64)
buf_x_out = cl.Buffer(ctx, mf.WRITE_ONLY, x_out_np.nbytes)
buf_y_out = cl.Buffer(ctx, mf.WRITE_ONLY, y_out_np.nbytes)

# local mem (batch affine)
local_size_bytes = WG_SIZE * 4 * 8   # WG * 4 ulongs * 8B
linvTot_size     = 4 * 8

# -------------------------
# Bench helper
# -------------------------
kernels = {
    "k_mul_batch_affine": cl.Kernel(prg, "k_mul_batch_affine"),
    "k_mul_only":         cl.Kernel(prg, "k_mul_only"),
    "k_xy64_to_affine":   cl.Kernel(prg, "k_xy64_to_affine"),  # <<< FIX
    "k_add_only":         cl.Kernel(prg, "k_add_only"),
    "k_add_batch_affine": cl.Kernel(prg, "k_add_batch_affine"),
    "k_add_indiv_affine": cl.Kernel(prg, "k_add_indiv_affine"),
}


def run_bench(title, k: cl.Kernel, gsz, lsz, args, ops_per_item=1):
    times = []
    for r in range(NUM_RUNS):
        # seta args (evita overhead de __call__ e evita retrieval)
        k.set_args(*args)

        evt = cl.enqueue_nd_range_kernel(queue, k, gsz, lsz)
        evt.wait()

        ms = (evt.profile.end - evt.profile.start) * 1e-6
        times.append(ms)

    meas = np.array(times[1:], dtype=np.float64)  # drop warm-up
    ops = NUM_ITEMS * ops_per_item
    thr = ops / (meas.mean() / 1000.0)

    print(f"\n=== {title} ===")
    print(f"Avg: {meas.mean():.3f} ms | Min: {meas.min():.3f} | Max: {meas.max():.3f} | Std: {meas.std():.3f}")
    print(f"Throughput: {thr:,.0f} ops/sec")


# -------------------------
# Run benches
# -------------------------
print(f"Device: {devs[0].name}")
print(f"NUM_ITEMS={NUM_ITEMS} WG={WG_SIZE} groups={NUM_ITEMS//WG_SIZE} runs={NUM_RUNS} (1 warm-up)")
print(f"ADD_ITERS={ADD_ITERS}")

gsz = (NUM_ITEMS,)
lsz = (WG_SIZE,)

# 1) mul + affine batch
run_bench(
    "1) point_mul + point_to_affine_batch64",
    kernels["k_mul_batch_affine"],
    gsz, lsz,
    (
        buf_k_be, buf_x_out, buf_y_out, buf_comb,
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(linvTot_size),
    ),
    ops_per_item=1
)

# 2) mul only
run_bench(
    "2) point_mul only (Jacobian)",
    kernels["k_mul_only"],
    gsz, lsz,
    (buf_k_be, buf_x_out, buf_y_out, buf_comb),
    ops_per_item=1
)

# 3) xy64_to_affine (individual affine)
run_bench(
    "3) xy64_to_affine (affine individual)",
    kernels["k_xy64_to_affine"],  # <<< FIX
    gsz, lsz,
    (buf_k_be, buf_x_out, buf_y_out, buf_comb),
    ops_per_item=1
)


# 4) point_add only
run_bench(
    "4) point_add_64 only (Jacobian)",
    kernels["k_add_only"],
    gsz, lsz,
    (buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS)),
    ops_per_item=ADD_ITERS
)

# 5) point_add + affine batch
run_bench(
    "5) point_add_64 + point_to_affine_batch64",
    kernels["k_add_batch_affine"],
    gsz, lsz,
    (
        buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(linvTot_size),
    ),
    ops_per_item=ADD_ITERS
)

# 6) point_add + affine individual
run_bench(
    "6) point_add_64 + point_to_affine (individual)",
    kernels["k_add_indiv_affine"],
    gsz, lsz,
    (buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS)),
    ops_per_item=ADD_ITERS
)
"""

full_src = common_src + math_src + point_src + bench_kernels
prg = cl.Program(ctx, full_src).build(options='-I .')

# -------------------------
# Buffers
# -------------------------
table_comb = comb.build_comb_table_u64()
buf_comb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=table_comb)

# random scalars (4x u64 por item)
rng = np.random.default_rng(12345)
k_words = rng.integers(0, np.iinfo(np.uint64).max, size=(NUM_ITEMS,4), dtype=np.uint64)
zmask = np.all(k_words == 0, axis=1)
k_words[zmask,3] = 1
k_be_np = np.ascontiguousarray(k_words.reshape(-1))
buf_k_be = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_be_np)

x_out_np = np.empty(NUM_ITEMS*4, dtype=np.uint64)
y_out_np = np.empty(NUM_ITEMS*4, dtype=np.uint64)
buf_x_out = cl.Buffer(ctx, mf.WRITE_ONLY, x_out_np.nbytes)
buf_y_out = cl.Buffer(ctx, mf.WRITE_ONLY, y_out_np.nbytes)

# local mem (batch affine)
local_size_bytes = WG_SIZE * 4 * 8   # WG * 4 ulongs * 8B
linvTot_size     = 4 * 8

# -------------------------
# Bench helper
# -------------------------
kernels = {
    "k_mul_batch_affine": cl.Kernel(prg, "k_mul_batch_affine"),
    "k_mul_only":         cl.Kernel(prg, "k_mul_only"),
    "k_xy64_to_affine":   cl.Kernel(prg, "k_xy64_to_affine"),  # <<< FIX
    "k_add_only":         cl.Kernel(prg, "k_add_only"),
    "k_add_batch_affine": cl.Kernel(prg, "k_add_batch_affine"),
    "k_add_indiv_affine": cl.Kernel(prg, "k_add_indiv_affine"),
}


def run_bench(title, k: cl.Kernel, gsz, lsz, args, ops_per_item=1):
    times = []
    for r in range(NUM_RUNS):
        # seta args (evita overhead de __call__ e evita retrieval)
        k.set_args(*args)

        evt = cl.enqueue_nd_range_kernel(queue, k, gsz, lsz)
        evt.wait()

        ms = (evt.profile.end - evt.profile.start) * 1e-6
        times.append(ms)

    meas = np.array(times[1:], dtype=np.float64)  # drop warm-up
    ops = NUM_ITEMS * ops_per_item
    thr = ops / (meas.mean() / 1000.0)

    print(f"\n=== {title} ===")
    print(f"Avg: {meas.mean():.3f} ms | Min: {meas.min():.3f} | Max: {meas.max():.3f} | Std: {meas.std():.3f}")
    print(f"Throughput: {thr:,.0f} ops/sec")


# -------------------------
# Run benches
# -------------------------
print(f"Device: {devs[0].name}")
print(f"NUM_ITEMS={NUM_ITEMS} WG={WG_SIZE} groups={NUM_ITEMS//WG_SIZE} runs={NUM_RUNS} (1 warm-up)")
print(f"ADD_ITERS={ADD_ITERS}")

gsz = (NUM_ITEMS,)
lsz = (WG_SIZE,)

# 1) mul + affine batch
run_bench(
    "1) point_mul + point_to_affine_batch64",
    kernels["k_mul_batch_affine"],
    gsz, lsz,
    (
        buf_k_be, buf_x_out, buf_y_out, buf_comb,
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(linvTot_size),
    ),
    ops_per_item=1
)

# 2) mul only
run_bench(
    "2) point_mul only (Jacobian)",
    kernels["k_mul_only"],
    gsz, lsz,
    (buf_k_be, buf_x_out, buf_y_out, buf_comb),
    ops_per_item=1
)

# 3) xy64_to_affine (individual affine)
run_bench(
    "3) xy64_to_affine (affine individual)",
    kernels["k_xy64_to_affine"],  # <<< FIX
    gsz, lsz,
    (buf_k_be, buf_x_out, buf_y_out, buf_comb),
    ops_per_item=1
)


# 4) point_add only
run_bench(
    "4) point_add_64 only (Jacobian)",
    kernels["k_add_only"],
    gsz, lsz,
    (buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS)),
    ops_per_item=ADD_ITERS
)

# 5) point_add + affine batch
run_bench(
    "5) point_add_64 + point_to_affine_batch64",
    kernels["k_add_batch_affine"],
    gsz, lsz,
    (
        buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(local_size_bytes),
        cl.LocalMemory(linvTot_size),
    ),
    ops_per_item=ADD_ITERS
)

# 6) point_add + affine individual
run_bench(
    "6) point_add_64 + point_to_affine (individual)",
    kernels["k_add_indiv_affine"],
    gsz, lsz,
    (buf_x_out, buf_y_out, buf_comb, np.uint32(ADD_ITERS)),
    ops_per_item=ADD_ITERS
)
```


### Step 2: Derive BIP-84
Example to invoke the derivation, use a kernel entry point that sets up variables and calls the derivation sequence.

```opencl
ulong index  = 0;
ulong change = 0;

change &= 1u;
index  &= 0x7fffffffu;

ulong M0, M1, M2, M3, M4t;
ulong I[8] = {0};

ulong k84[4], c84[4];
ulong k84_0[4], c84_0[4];
ulong k84_0_0[4], c84_0_0[4];

ulong kchg[4], cchg[4];
ulong k_out[4], c_out[4];

// locals do batch affine (1x por WG)
__local ulong lz[WG*4];
__local ulong lprefix[WG*4];
__local ulong lscan[WG*4];
__local ulong linvTot[4];

// m/84' (hardened)
if (!ckd_priv_hardened(km, cm, 84u, k84, c84, &M0,&M1,&M2,&M3,&M4t, I)) return;
// m/84'/0' (hardened)
if (!ckd_priv_hardened(k84, c84, 0u, k84_0, c84_0, &M0,&M1,&M2,&M3,&M4t, I)) return;
// m/84'/0'/0' (hardened)
if (!ckd_priv_hardened(k84_0, c84_0, 0u, k84_0_0, c84_0_0, &M0,&M1,&M2,&M3,&M4t, I)) return;

// pub m/84'/0'/0'  (X,Y affine via batch-only)
ulong Xacc64[4], Yacc64[4];
pub_affine_batch64(Xacc64, Yacc64, k84_0_0, COMB, lz, lprefix, lscan, linvTot);

// m/84'/0'/0'/change  (normal usando X/Y)
if (!ckd_priv_normal_from_xy(k84_0_0, c84_0_0, Xacc64, Yacc64, (uint)change,
                            kchg, cchg, &M0,&M1,&M2,&M3,&M4t, I)) return;

// pub m/84'/0'/0'/change (X,Y affine via batch-only)
ulong Xchg64[4], Ychg64[4];
pub_affine_batch64(Xchg64, Ychg64, kchg, COMB, lz, lprefix, lscan, linvTot);

// m/84'/0'/0'/change/index (normal)
if (!ckd_priv_normal_from_xy(kchg, cchg, Xchg64, Ychg64, (uint)index,
                            k_out, c_out, &M0,&M1,&M2,&M3,&M4t, I)) return;

// pub final (X,Y) via batch-only
ulong X64[4], Y64[4];
pub_affine_batch64(X64, Y64, k_out, COMB, lz, lprefix, lscan, linvTot);
```

## Building / Integration
This repo is primarily OpenCL kernel code. Typical integration steps:
1. Concatenate / include the `.cl` files in your host program.
2. Allocate buffers for: input scalars / chain codes / intermediate states; comb precomputed table buffer; output points / derived keys.
3. Dispatch kernels with a tuned global/local size.



## Limitations / Assumptions
- Designed for throughput, not minimal code size.
- Tables can be large depending on Comb parameters; tune to your VRAM and cache behavior.
- Side-channel guarantees are not the same as hardened CPU libraries.

This combination (ultrafast reduction + optimized squaring + Comb w=16 + mixed addition + batch inversion) is the current state of the art for secp256k1 on GPU. There is no known faster public implementation as of January 2026.
