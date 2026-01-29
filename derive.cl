

inline void pack_hardened37(const ulong k[4], uint i, __private ulong *M0,
                            __private ulong *M1, __private ulong *M2,
                            __private ulong *M3, __private ulong *M4t) {
  *M0 = (k[0] >> 8);
  *M1 = ((k[0] & 0xFFUL) << 56) | (k[1] >> 8);
  *M2 = ((k[1] & 0xFFUL) << 56) | (k[2] >> 8);
  *M3 = ((k[2] & 0xFFUL) << 56) | (k[3] >> 8);
  *M4t = ((k[3] & 0xFFUL) << 56) | ((ulong)((i >> 24) & 0xFF) << 48) |
         ((ulong)((i >> 16) & 0xFF) << 40) | ((ulong)((i >> 8) & 0xFF) << 32) |
         ((ulong)(i & 0xFF) << 24);
}

inline void pack_normal37_xy64_le4(const ulong X64[4], const ulong Y64[4],
                                   uint i, ulong *M0, ulong *M1, ulong *M2,
                                   ulong *M3, ulong *M4t) {
  const ulong Xbe0 = X64[3];
  const ulong Xbe1 = X64[2];
  const ulong Xbe2 = X64[1];
  const ulong Xbe3 = X64[0];
  const ulong pfx = ((Y64[0] & 1UL) == 0UL ? 0x02UL : 0x03UL) << 56;

  *M0 = pfx | (Xbe0 >> 8);
  *M1 = ((Xbe0 & 0xFFUL) << 56) | (Xbe1 >> 8);
  *M2 = ((Xbe1 & 0xFFUL) << 56) | (Xbe2 >> 8);
  *M3 = ((Xbe2 & 0xFFUL) << 56) | (Xbe3 >> 8);
  *M4t = ((Xbe3 & 0xFFUL) << 56) | ((ulong)((i >> 24) & 0xFF) << 48) |
         ((ulong)((i >> 16) & 0xFF) << 40) | ((ulong)((i >> 8) & 0xFF) << 32) |
         ((ulong)(i & 0xFF) << 24);
}

#ifndef B32SW_HARDENED
#define B32SW_HARDENED(i) ((uint)(0x80000000u | ((i) & 0x7fffffffu)))
#endif

// retorna 0 se key inválida (k==0), 1 se ok
static inline int ckd_priv_hardened(const ulong kpar[4], const ulong cpar[4],
                                    uint idx_hardened, ulong kout[4],
                                    ulong cout[4], __private ulong *M0,
                                    __private ulong *M1, __private ulong *M2,
                                    __private ulong *M3, __private ulong *M4t,
                                    __private ulong I[8]) {
  pack_hardened37(kpar, B32SW_HARDENED(idx_hardened), M0, M1, M2, M3, M4t);
  hmac512_ccode_msg37(cpar, *M0, *M1, *M2, *M3, *M4t, I);

  kout[0] = I[0];
  kout[1] = I[1];
  kout[2] = I[2];
  kout[3] = I[3];
  cout[0] = I[4];
  cout[1] = I[5];
  cout[2] = I[6];
  cout[3] = I[7];

  addmod_n(kout, kout, kpar);
  return ((kout[0] | kout[1] | kout[2] | kout[3]) != 0UL);
}

// retorna 0 se key inválida (k==0), 1 se ok
static inline int ckd_priv_normal_from_xy(
    const ulong kpar[4], const ulong cpar[4], const ulong X64[4],
    const ulong Y64[4], uint index, ulong kout[4], ulong cout[4],
    __private ulong *M0, __private ulong *M1, __private ulong *M2,
    __private ulong *M3, __private ulong *M4t, __private ulong I[8]) {
  // mensagem normal: (serP || index) já embutido no teu pack_normal37_xy64_le4
  pack_normal37_xy64_le4(X64, Y64, index, M0, M1, M2, M3, M4t);
  hmac512_ccode_msg37(cpar, *M0, *M1, *M2, *M3, *M4t, I);

  kout[0] = I[0];
  kout[1] = I[1];
  kout[2] = I[2];
  kout[3] = I[3];
  cout[0] = I[4];
  cout[1] = I[5];
  cout[2] = I[6];
  cout[3] = I[7];

  addmod_n(kout, kout, kpar);
  return ((kout[0] | kout[1] | kout[2] | kout[3]) != 0UL);
}

// pubkey (affine) usando somente batch64 (1 inv por WG)
static inline void pub_affine_batch64(ulong X64[4], ulong Y64[4],
                                      const ulong k_be[4],
                                      __global const ulong *COMB,
                                      __local ulong *lz, __local ulong *lprefix,
                                      __local ulong *lscan,
                                      __local ulong *linvTot) {
  ulong k_le64[4];
  scalar_be64_to_le64_4(k_be, k_le64);

  jac_t R;
  point_mulG_comb8(&R, k_le64, COMB);
  int inf = point_is_inf(&R);

  point_to_affine_batch64(X64, Y64, &R, inf, lz, lprefix, lscan, linvTot);
}
