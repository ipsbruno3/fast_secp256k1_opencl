

#define COMB_W 16
#define COMB_D 16
#define COMB_T (1u << COMB_W)

typedef ulong fe_t[4];
typedef struct {
  fe_t x, y, z;
} jac_t;

  
inline void point_set_inf(__private jac_t *P) {
  fe_clear(P->x);
  fe_clear(P->y);
  fe_clear(P->z);
}
inline int point_is_inf(const __private jac_t *P) { 
  return fe_is_zero(P->z);
}

inline void point_set_affine(__private jac_t *P, const __private ulong ax[4],
                             const __private ulong ay[4]) {
  COPY_FOUR(P->x, ax);
  COPY_FOUR(P->y, ay);
  fe_set_one(P->z);
}

inline void point_copy(__private jac_t *R, const __private jac_t *P) {
  fe_copy(R->x, P->x);
  fe_copy(R->y, P->y);
  fe_copy(R->z, P->z);
}

inline void point_double64(__private jac_t *R, const __private jac_t *P) {
  if (point_is_inf(P) || fe_is_zero(P->y)) {
    point_set_inf(R);
    return;
  }
  ulong A_[4], B_[4], C_[4], D_[4], E_[4], F_[4], tmp[4], tmp2[4];
  fe_sqr(A_, P->x);
  fe_sqr(B_, P->y);
  fe_sqr(C_, B_);
  fe_add(tmp, P->x, B_);
  fe_sqr(tmp, tmp);
  fe_sub(tmp, tmp, A_);
  fe_sub(tmp, tmp, C_);
  fe_add(D_, tmp, tmp);
  fe_add(tmp, A_, A_);
  fe_add(E_, tmp, A_);
  fe_sqr(F_, E_);
  fe_add(tmp, D_, D_);
  fe_sub(R->x, F_, tmp);
  fe_sub(tmp, D_, R->x);
  fe_mul(tmp, E_, tmp);
  fe_add(tmp2, C_, C_);
  fe_add(tmp2, tmp2, tmp2);
  fe_add(tmp2, tmp2, tmp2);
  fe_sub(R->y, tmp, tmp2);
  fe_mul(tmp, P->y, P->z);
  fe_add(R->z, tmp, tmp);
}

inline void point_add_64(__private jac_t *R, const __private jac_t *P, const __private jac_t *Q) {
  if (point_is_inf(P)) {
    point_copy(R, Q);
    return;
  }
  if (point_is_inf(Q)) {
    point_copy(R, P);
    return;
  }
  ulong Z1Z1[4], Z2Z2[4], U1[4], U2[4], S1[4], S2[4];
  ulong H[4], RR[4], HH[4], HHH[4], V[4], tmp[4], tmp2[4];
  fe_sqr(Z1Z1, P->z);
  fe_sqr(Z2Z2, Q->z);
  fe_mul(U1, P->x, Z2Z2);
  fe_mul(U2, Q->x, Z1Z1);
  fe_mul(tmp, Q->z, Z2Z2);
  fe_mul(S1, P->y, tmp);
  fe_mul(tmp, P->z, Z1Z1);
  fe_mul(S2, Q->y, tmp);
  fe_sub(H, U2, U1);
  fe_sub(RR, S2, S1);
  if (fe_is_zero(H)) {
    if (fe_is_zero(RR))
      point_double64(R, P);
    else {
      point_set_inf(R);
    }
    return;
  }
  fe_sqr(HH, H);
  fe_mul(HHH, H, HH);
  fe_mul(V, U1, HH);
  fe_sqr(tmp, RR);
  fe_add(tmp2, V, V);
  fe_sub(tmp, tmp, HHH);
  fe_sub(R->x, tmp, tmp2);
  fe_sub(tmp, V, R->x);
  fe_mul(tmp, RR, tmp);
  fe_mul(tmp2, S1, HHH);
  fe_sub(R->y, tmp, tmp2);
  fe_mul(tmp, P->z, Q->z);
  fe_mul(R->z, H, tmp);
}

inline void scalar_be64_to_le64_4(const ulong k_be[4], ulong k_le[4]) {
  k_le[0] = k_be[3];  k_le[1] = k_be[2];  k_le[2] = k_be[1];  k_le[3] = k_be[0];
}

inline void le64_to_le32_8(const ulong in64[4], uint out32[8]) {
  #pragma unroll
  for (int i=0;i<4;i++){
    ulong v = in64[i];
    out32[i*2 + 0] = (uint)(v);
    out32[i*2 + 1] = (uint)(v >> 32);
  }
}

inline void point_to_affine(__private ulong ax[4], __private ulong ay[4],
                            const __private jac_t *P) {
  if (point_is_inf(P)) {
    fe_clear(ax);
    fe_clear(ay);
    return;
  }
  ulong zinv[4], z2[4], z3[4];
  fe_inv(zinv, P->z);
  fe_sqr(z2, zinv);
  fe_mul(z3, z2, zinv);
  fe_mul(ax, P->x, z2);
  fe_mul(ay, P->y, z3);
}

static inline void point_double64_inplace(__private jac_t *P) {
  jac_t R;
  point_double64(&R, P);
  point_copy(P, &R);
}

inline void point_add_affine(__private jac_t *R, const __private jac_t *P,
                             const __private ulong Qx[4], const __private ulong Qy[4]) {
  if (point_is_inf(P)) {
    point_set_affine(R, Qx, Qy);
    return;
  }
  ulong Z1Z1[4], U2[4], S2[4], H[4], RR[4], HH[4], HHH[4], V[4];
  ulong tmp[4], tmp2[4];
  fe_sqr(Z1Z1, P->z);
  fe_mul(U2, Qx, Z1Z1);
  fe_mul(tmp, P->z, Z1Z1);
  fe_mul(S2, Qy, tmp);
  fe_sub(H, U2, P->x);
  fe_sub(RR, S2, P->y); 
  if (fe_is_zero(H)) {
    if (fe_is_zero(RR)) {
      point_double64(R, P);
    } else {
      point_set_inf(R);
    }
    return;
  }
  fe_sqr(HH, H);
  fe_mul(HHH, H, HH);
  fe_mul(V, P->x, HH);
  fe_sqr(tmp, RR);
  fe_add(tmp2, V, V);
  fe_sub(tmp, tmp, HHH);
  fe_sub(R->x, tmp, tmp2);
  fe_sub(tmp, V, R->x);
  fe_mul(tmp, RR, tmp);
  fe_mul(tmp2, P->y, HHH);
  fe_sub(R->y, tmp, tmp2);
  fe_mul(R->z, P->z, H);
}



static inline void point_add_affine_global_inplace(__private jac_t *P,
                                                   __global const ulong *p) {
  jac_t R;
  ulong Qx[4] = {p[0], p[1], p[2], p[3]};
  ulong Qy[4] = {p[4], p[5], p[6], p[7]};
  point_add_affine(&R, P, Qx, Qy);
  point_copy(P, &R);
}

static inline uint comb16_nib(ulong kw, uint col) {
  ulong x = (kw >> col) & 0x0001000100010001UL;
  return (uint)((x | (x >> 15) | (x >> 30) | (x >> 45)) & 0xFu);
}

static inline uint comb16_idx(const ulong k0, const ulong k1,
                              const ulong k2, const ulong k3, uint col) {
  return  comb16_nib(k0, col)
       | (comb16_nib(k1, col) << 4)
       | (comb16_nib(k2, col) << 8)
       | (comb16_nib(k3, col) << 12);
}
inline void point_mulG_comb8(__private jac_t *R,
                                 const __private ulong k[4],
                                 __global const ulong *restrict COMB) {
  jac_t acc;
  point_set_inf(&acc);
  const ulong k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint col = 15u - (uint)i;
    point_double64_inplace(&acc);
    const uint idx = comb16_idx(k0, k1, k2, k3, col);
    if (idx) {
      const __global ulong *p = COMB + ((size_t)idx << 3);  // idx * 8 ulongs
      point_add_affine_global_inplace(&acc, p);
    }
  }
  point_copy(R, &acc);
}

static inline void point_to_affine_batch64(
    __private ulong ax[4], __private ulong ay[4],
    const __private jac_t *P, int is_inf,
    __local ulong *lz,       // WG*4
    __local ulong *lprefix,  // WG*4
    __local ulong *lscan,    // WG*4 (reusado)
    __local ulong *linvTot)  // 4
{
  const uint lid = get_local_id(0);
  const uint ridx = (WG - 1u) - lid;

  // z_or1 = (is_inf ? 1 : P->z)
  ulong z_or1[4];
  if (is_inf) fe_one(z_or1); else fe_copy(z_or1, P->z);

  // guarda z original e prepara scan
  fe_store_local(lz,    lid, z_or1);
  fe_store_local(lscan, lid, z_or1);
  barrier(CLK_LOCAL_MEM_FENCE);

  // prefix exclusivo
  blelloch_scan_excl_mul_fe64(lscan);

  ulong prefix[4];
  fe_load_local(prefix, lscan, lid);
  fe_store_local(lprefix, lid, prefix);
  barrier(CLK_LOCAL_MEM_FENCE);

  // invTotal = inv(prod Z)
  if (lid == 0) {
    ulong preLast[4], zLast[4], total[4], invT[4];
    fe_load_local(preLast, lprefix, WG - 1u); // prod Z[0..62]
    fe_load_local(zLast,   lz,      WG - 1u); // Z[63] (ou 1)
    fe_mul(total, preLast, zLast);           // total = prod Z[0..63]
    fe_inv(invT, total);

    linvTot[0] = invT[0]; linvTot[1] = invT[1];
    linvTot[2] = invT[2]; linvTot[3] = invT[3];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // prepara scan no vetor reverso (pra suffix exclusivo)
  ulong zrev[4];
  fe_load_local(zrev, lz, ridx);   // z[WG-1-lid]
  fe_store_local(lscan, lid, zrev);
  barrier(CLK_LOCAL_MEM_FENCE);

  blelloch_scan_excl_mul_fe64(lscan);
  barrier(CLK_LOCAL_MEM_FENCE);

  // suffix_excl[i] = rev_prefix_excl[WG-1-i]
  ulong suffix[4];
  fe_load_local(suffix, lscan, ridx);

  // invZ = invTotal * prefix * suffix
  ulong invTotal[4] = { linvTot[0], linvTot[1], linvTot[2], linvTot[3] };
  ulong tmp[4], invZ[4];
  fe_mul(tmp, prefix, suffix);
  fe_mul(invZ, tmp, invTotal);

  // mascara pontos no infinito
  if (is_inf) fe_clear(invZ);

  // ax = X * invZ^2 ; ay = Y * invZ^3
  ulong z2[4], z3[4];
  fe_sqr(z2, invZ);
  fe_mul(z3, z2, invZ);
  fe_mul(ax, P->x, z2);
  fe_mul(ay, P->y, z3);
}

inline void xy64_to_affine(ulong X64[4], ulong Y64[4],
                                      const ulong k_be[4],
                                      __global const ulong *COMB)
{
  ulong k_le64[4];
  scalar_be64_to_le64_4(k_be, k_le64);   
  jac_t R;
  point_mulG_comb8(&R, k_le64, COMB);
  point_to_affine(X64, Y64, &R);
}
