// ===== n (ordem do grupo) em 4x64 big-endian =====
#define N0 ((ulong)0xFFFFFFFFFFFFFFFFUL)
#define N1 ((ulong)0xFFFFFFFFFFFFFFFEUL)
#define N2 ((ulong)0xBAAEDCE6AF48A03BUL)
#define N3 ((ulong)0xBFD25E8CD0364141UL)

__constant const ulong SECP_P[4] = {
    (ulong)0xFFFFFFFEFFFFFC2FUL, (ulong)0xFFFFFFFFFFFFFFFFUL,
    (ulong)0xFFFFFFFFFFFFFFFFUL, (ulong)0xFFFFFFFFFFFFFFFFUL};

__constant const ulong SECP_GX[4] = {
    (ulong)0x59F2815B16F81798UL, (ulong)0x029BFCDB2DCE28D9UL,
    (ulong)0x55A06295CE870B07UL, (ulong)0x79BE667EF9DCBBACUL};

__constant const ulong SECP_GY[4] = {
    (ulong)0x9C47D08FFB10D4B8UL, (ulong)0xFD17B448A6855419UL,
    (ulong)0x5DA4FBFC0E1108A8UL, (ulong)0x483ADA7726A3C465UL};

inline ulong addc64(ulong x, ulong y, __private ulong *c) {
  ulong s = x + y;
  ulong c1 = (s < x);
  ulong s2 = s + *c;
  ulong c2 = (s2 < s);
  *c = (c1 | c2);
  return s2;
}

inline ulong subb64(ulong x, ulong y, __private ulong *b) {
  ulong t = x - y;
  ulong b1 = (x < y);
  ulong t2 = t - *b;
  ulong b2 = (t < *b);
  *b = (b1 | b2);
  return t2;
}

inline void fe_copy(__private ulong r[4], const __private ulong a[4]) {
  r[0] = a[0];
  r[1] = a[1];
  r[2] = a[2];
  r[3] = a[3];
}
inline void fe_clear(__private ulong r[4]) {
  r[0] = 0;
  r[1] = 0;
  r[2] = 0;
  r[3] = 0;
}
inline void fe_set_one(__private ulong r[4]) {
  r[0] = 1;
  r[1] = 0;
  r[2] = 0;
  r[3] = 0;
}

inline int fe_is_zero(const __private ulong a[4]) {
  return (a[0] | a[1] | a[2] | a[3]) == 0;
}


inline void fe_cond_sub_p(__private ulong a[4]) {
  ulong br=0;
  ulong t0=subb64(a[0], SECP_P[0], &br);
  ulong t1=subb64(a[1], SECP_P[1], &br);
  ulong t2=subb64(a[2], SECP_P[2], &br);
  ulong t3=subb64(a[3], SECP_P[3], &br);

  ulong m = (ulong)0 - (ulong)(br==0); // all-ones se a>=p
  a[0] = bitselect(a[0], t0, m);
  a[1] = bitselect(a[1], t1, m);
  a[2] = bitselect(a[2], t2, m);
  a[3] = bitselect(a[3], t3, m);
}

inline void fe_add(__private ulong r[4], const __private ulong a[4], const __private ulong b[4]) {
  ulong c=0;
  ulong s0=addc64(a[0], b[0], &c);
  ulong s1=addc64(a[1], b[1], &c);
  ulong s2=addc64(a[2], b[2], &c);
  ulong s3=addc64(a[3], b[3], &c);

  ulong br=0;
  ulong t0=subb64(s0, SECP_P[0], &br);
  ulong t1=subb64(s1, SECP_P[1], &br);
  ulong t2=subb64(s2, SECP_P[2], &br);
  ulong t3=subb64(s3, SECP_P[3], &br);

  ulong use_t = (ulong)(c | (br==0)); // c==1 ou s>=p
  ulong m = (ulong)0 - use_t;
  r[0] = bitselect(s0, t0, m);
  r[1] = bitselect(s1, t1, m);
  r[2] = bitselect(s2, t2, m);
  r[3] = bitselect(s3, t3, m);
}

inline void fe_sub(__private ulong r[4], const __private ulong a[4], const __private ulong b[4]) {
  ulong br=0;
  ulong d0=subb64(a[0], b[0], &br);
  ulong d1=subb64(a[1], b[1], &br);
  ulong d2=subb64(a[2], b[2], &br);
  ulong d3=subb64(a[3], b[3], &br);

  ulong c=0;
  ulong ap0=addc64(d0, SECP_P[0], &c);
  ulong ap1=addc64(d1, SECP_P[1], &c);
  ulong ap2=addc64(d2, SECP_P[2], &c);
  ulong ap3=addc64(d3, SECP_P[3], &c);

  ulong m = (ulong)0 - br; // se borrow, escolhe ap
  r[0] = bitselect(d0, ap0, m);
  r[1] = bitselect(d1, ap1, m);
  r[2] = bitselect(d2, ap2, m);
  r[3] = bitselect(d3, ap3, m);
}


inline void acc_add128(__private ulong *lo, __private ulong *hi,
                       __private ulong *ex, ulong plo, ulong phi) {
  ulong nlo = *lo + plo;
  ulong c0 = (nlo < *lo);
  *lo = nlo;
  ulong t = *hi + phi;
  ulong cA = (t < *hi);
  ulong t2 = t + c0;
  ulong cB = (t2 < t);
  *hi = t2;
  *ex += (cA + cB);
}

inline void mul256_raw(__private ulong t[8], const __private ulong a[4],
                       const __private ulong b[4]) {
  const ulong a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const ulong b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

  ulong carry_lo = 0, carry_hi = 0;
  ulong lo, hi, ex;

#define MAC(x, y)                                                              \
  do {                                                                         \
    ulong plo = (x) * (y);                                                     \
    ulong phi = mul_hi((x), (y));                                              \
    acc_add128(&lo, &hi, &ex, plo, phi);                                       \
  } while (0)
  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a0, b0);
  t[0] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a0, b1);
  MAC(a1, b0);
  t[1] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a0, b2);
  MAC(a1, b1);
  MAC(a2, b0);
  t[2] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a0, b3);
  MAC(a1, b2);
  MAC(a2, b1);
  MAC(a3, b0);
  t[3] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a1, b3);
  MAC(a2, b2);
  MAC(a3, b1);
  t[4] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a2, b3);
  MAC(a3, b2);
  t[5] = lo;
  carry_lo = hi;
  carry_hi = ex;

  lo = carry_lo;
  hi = carry_hi;
  ex = 0;
  MAC(a3, b3);
  t[6] = lo;
  carry_lo = hi;
  carry_hi = ex;
  t[7] = carry_lo;

#undef MAC
}

inline ulong add64_cc(ulong s, ulong x, __private uint *cc) {
  ulong t = s + x;
  *cc += (t < s);
  return t;
}

inline void fe_reduce512(__private ulong r[4], const __private ulong t[8]) {
  ulong H0_A = t[4], H1_A = t[5], H2_A = t[6], H3_A = t[7];
  ulong m0_hi = mul_hi(H0_A, 977UL);
  ulong m1_hi = mul_hi(H1_A, 977UL);
  ulong m2_hi = mul_hi(H2_A, 977UL);
  ulong m3_hi = mul_hi(H3_A, 977UL);

  ulong s0_lo = (H0_A << 32), s0_hi = (H0_A >> 32);
  ulong s1_lo = (H1_A << 32), s1_hi = (H1_A >> 32);
  ulong s2_lo = (H2_A << 32), s2_hi = (H2_A >> 32);
  ulong s3_lo = (H3_A << 32), s3_hi = (H3_A >> 32);

  ulong r5[5];
  ulong carry = 0;
  uint cc = 0;
  ulong s = t[0];
  {

    s = add64_cc(s, H0_A * 977UL, &cc);
    s = add64_cc(s, s0_lo, &cc);
    s = add64_cc(s, carry, &cc);
    r5[0] = s;
    carry = (ulong)cc;
  }
  {
    cc = 0;
    s = t[1];
    s = add64_cc(s, H1_A * 977UL, &cc);
    s = add64_cc(s, m0_hi, &cc);
    s = add64_cc(s, s1_lo, &cc);
    s = add64_cc(s, s0_hi, &cc);
    s = add64_cc(s, carry, &cc);
    r5[1] = s;
    carry = (ulong)cc;
  }
  {
    cc = 0;
    s = t[2];
    s = add64_cc(s, H2_A * 977UL, &cc);
    s = add64_cc(s, m1_hi, &cc);
    s = add64_cc(s, s2_lo, &cc);
    s = add64_cc(s, s1_hi, &cc);
    s = add64_cc(s, carry, &cc);
    r5[2] = s;
    carry = (ulong)cc;
  }
  {
    cc = 0;
    s = t[3];
    s = add64_cc(s, H3_A * 977UL, &cc);
    s = add64_cc(s, m2_hi, &cc);
    s = add64_cc(s, s3_lo, &cc);
    s = add64_cc(s, s2_hi, &cc);
    s = add64_cc(s, carry, &cc);
    r5[3] = s;
    carry = (ulong)cc;
  }
  {
    cc = 0;
    s = 0;
    s = add64_cc(s, m3_hi, &cc);
    s = add64_cc(s, s3_hi, &cc);
    s = add64_cc(s, carry, &cc);
    r5[4] = s;
  }

#pragma unroll
  for (int it = 0; it < 3; ++it) {
    ulong k = r5[4];
    if (k == 0)
      break;
    r5[4] = 0;
    ulong k_lo = k * 977UL, k_hi = mul_hi(k, 977UL);
    ulong k_slo = (k << 32), k_shi = (k >> 32);
    ulong c = 0;
    {
      cc = 0;
      s = r5[0];
      s = add64_cc(s, k_lo, &cc);
      s = add64_cc(s, k_slo, &cc);
      s = add64_cc(s, c, &cc);
      r5[0] = s;
      c = (ulong)cc;
    }
    {
      cc = 0;
      s = r5[1];
      s = add64_cc(s, k_hi, &cc);
      s = add64_cc(s, k_shi, &cc);
      s = add64_cc(s, c, &cc);
      r5[1] = s;
      c = (ulong)cc;
    }
#pragma unroll
    for (int i = 2; i < 5; i++) {
      cc = 0;
      s = r5[i];
      s = add64_cc(s, c, &cc);
      r5[i] = s;
      c = (ulong)cc;
    }
  }

  r[0] = r5[0];
  r[1] = r5[1];
  r[2] = r5[2];
  r[3] = r5[3];
  fe_cond_sub_p(r);
  fe_cond_sub_p(r);
}

inline void fe_mul(__private ulong r[4], const __private ulong a[4],
                   const __private ulong b[4]) {
  ulong t[8];
  mul256_raw(t, a, b);
  fe_reduce512(r, t);
}

inline void sqr256_raw(__private ulong t[8], const __private ulong a[4]) {
  const ulong a0=a[0], a1=a[1], a2=a[2], a3=a[3];
  ulong carry_lo=0, carry_hi=0;
  ulong lo, hi, ex;
  #define MAC1(x,y) do{ \
    ulong plo=(x)*(y); \
    ulong phi=mul_hi((x),(y)); \
    acc_add128(&lo,&hi,&ex, plo, phi); \
  } while(0)
  #define MAC2(x,y) do{ \
    ulong plo=(x)*(y); \
    ulong phi=mul_hi((x),(y)); \
    ulong plo2 = plo + plo; \
    ulong c0  = (plo2 < plo); \
    ulong phi2 = phi + phi; \
    ulong c1  = (phi2 < phi); \
    ulong phi3 = phi2 + c0; \
    ulong c2  = (phi3 < phi2); \
    acc_add128(&lo,&hi,&ex, plo2, phi3); \
    ex += (c1 + c2); \
  } while(0)
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC1(a0,a0);
  t[0]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC2(a0,a1);
  t[1]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC2(a0,a2);
  MAC1(a1,a1);
  t[2]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC2(a0,a3);
  MAC2(a1,a2);
  t[3]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC2(a1,a3);
  MAC1(a2,a2);
  t[4]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC2(a2,a3);
  t[5]=lo; carry_lo=hi; carry_hi=ex;
  lo=carry_lo; hi=carry_hi; ex=0;
  MAC1(a3,a3);
  t[6]=lo; carry_lo=hi; carry_hi=ex;
  t[7]=carry_lo;
  #undef MAC1
  #undef MAC2
}

inline void fe_sqr(__private ulong r[4], const __private ulong a[4]) {
  ulong t[8];
  sqr256_raw(t, a);
  fe_reduce512(r, t);
}



inline void add256_be(__private ulong r[4], const __private ulong a[4],
                      const __private ulong b[4], __private int *carry) {
  ulong c = 0;
  for (int i = 3; i >= 0; --i) {
    ulong s = a[i] + b[i], s2 = s + c;
    r[i] = s2;
    c = (s < b[i]) | (s2 < s);
  }
  *carry = (int)c;
}


inline void sub256_be(__private ulong r[4], const __private ulong a[4],
                      const __private ulong b[4]) {
  ulong c = 0;
  for (int i = 3; i >= 0; --i) {
    ulong bi = b[i] + c;
    c = (a[i] < bi);
    r[i] = a[i] - bi;
  }
}

inline int ge_n(const __private ulong a[4]) {
  if (a[0] != N0)
    return a[0] > N0;
  if (a[1] != N1)
    return a[1] > N1;
  if (a[2] != N2)
    return a[2] > N2;
  return a[3] >= N3;
}

inline void addmod_n(__private ulong r[4], const __private ulong a[4],
                     const __private ulong b[4]) {
  int carry;
  add256_be(r, a, b, &carry);
  if (carry || ge_n(r)) {
    const __private ulong N_BE[4] = {N0, N1, N2, N3};
    sub256_be(r, r, N_BE);
  }
}

#define FE_SQRN_INPLACE(v, n)                  \
  do {                                         \
    _Pragma("unroll")                          \
    for (int _i = 0; _i < (n); ++_i) {         \
      fe_sqr((v), (v));                        \
    }                                          \
  } while (0)

inline void fe_inv(__private ulong r[4], const __private ulong a[4]) {
  if (fe_is_zero(a)) {
    fe_clear(r);
    return;
  }

  // x1 = a
  ulong x1[4]; fe_copy(x1, a);

  // x2 = a^(2^2 - 1) = a^3
  ulong x2[4], t[4];
  fe_sqr(t, x1);       // a^2
  fe_mul(x2, t, x1);   // a^3

  // x3 = a^(2^3 - 1) = a^7
  ulong x3[4];
  fe_sqr(t, x2);       // a^6
  fe_mul(x3, t, x1);   // a^7

  // x11 = a^(2^11 - 1)
  // Começa de x3 (=2^3-1), e vai “estendendo bloco de 1s”
  ulong x11[4];
  fe_copy(t, x3);
  FE_SQRN_INPLACE(t, 3);   fe_mul(t, t, x3);   // 2^6 - 1
  FE_SQRN_INPLACE(t, 3);   fe_mul(t, t, x3);   // 2^9 - 1
  FE_SQRN_INPLACE(t, 2);   fe_mul(x11, t, x2); // 2^11 - 1

  // x22 = a^(2^22 - 1)
  ulong x22[4];
  fe_copy(t, x11);
  FE_SQRN_INPLACE(t, 11);
  fe_mul(x22, t, x11);

  // x44 = a^(2^44 - 1)
  ulong x44[4];
  fe_copy(t, x22);
  FE_SQRN_INPLACE(t, 22);
  fe_mul(x44, t, x22);

  // x88 = a^(2^88 - 1)
  ulong x88[4];
  fe_copy(t, x44);
  FE_SQRN_INPLACE(t, 44);
  fe_mul(x88, t, x44);

  // Agora “monta” o expoente p-3 (a^-2), partindo de x88:
  fe_copy(t, x88);

  FE_SQRN_INPLACE(t, 88);  fe_mul(t, t, x88);
  FE_SQRN_INPLACE(t, 44);  fe_mul(t, t, x44);
  FE_SQRN_INPLACE(t, 3);   fe_mul(t, t, x3);
  FE_SQRN_INPLACE(t, 23);  fe_mul(t, t, x22);
  FE_SQRN_INPLACE(t, 5);   fe_mul(t, t, x1);
  FE_SQRN_INPLACE(t, 3);   fe_mul(t, t, x2);
  FE_SQRN_INPLACE(t, 2);   // termina o q-3

  // t = a^(p-3) = a^-2  =>  r = t * a = a^-1
  fe_mul(r, t, x1);
}

#undef FE_SQRN_INPLACE

#define WG 64u

inline void fe_store_local(__local ulong *dst, uint idx, const __private ulong a[4]) {
  uint o = idx * 4u;
  dst[o+0] = a[0]; dst[o+1] = a[1]; dst[o+2] = a[2]; dst[o+3] = a[3];
}
inline void fe_load_local(__private ulong a[4], __local const ulong *src, uint idx) {
  uint o = idx * 4u;
  a[0] = src[o+0]; a[1] = src[o+1]; a[2] = src[o+2]; a[3] = src[o+3];
}
inline void fe_one(__private ulong a[4]) { a[0]=1; a[1]=0; a[2]=0; a[3]=0; }

// scan: em `scan[idx]` (cada elemento tem 4 ulongs), faz exclusive prefix-product
static inline void blelloch_scan_excl_mul_fe64(__local ulong *scan) {
  const uint lid = get_local_id(0);

  // Up-sweep
  #pragma unroll
  for (uint offset = 1; offset < WG; offset <<= 1) {
    uint idx = ((lid + 1u) * (offset << 1)) - 1u;
    if (idx < WG) {
      ulong A[4], B[4], T[4];
      fe_load_local(A, scan, idx - offset);
      fe_load_local(B, scan, idx);
      fe_mul(T, A, B);                 // T = A*B
      fe_store_local(scan, idx, T);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Set root = 1
  if (lid == 0) {
    ulong one[4]; fe_one(one);
    fe_store_local(scan, WG - 1u, one);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Down-sweep
  #pragma unroll
  for (uint offset = (WG >> 1); offset > 0; offset >>= 1) {
    uint idx = ((lid + 1u) * (offset << 1)) - 1u;
    if (idx < WG) {
      ulong t[4], s[4], new_idx[4];
      fe_load_local(t, scan, idx - offset);
      fe_load_local(s, scan, idx);

      // scan[idx-offset] = s
      fe_store_local(scan, idx - offset, s);

      // scan[idx] = s * t  (ordem não importa no campo)
      fe_mul(new_idx, s, t);
      fe_store_local(scan, idx, new_idx);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
