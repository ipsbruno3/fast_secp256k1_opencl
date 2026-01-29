#define IPAD 0x3636363636363636UL
#define OPAD 0x5c5c5c5c5c5c5c5cUL

#define BITCOIN_SEED 0x426974636f696e20UL, 0x7365656400000000UL, 0, 0
#define SEED_VERSION  0x5365656420766572ULL, 0x73696f6e00000000ULL, 0ULL, 0ULL

#define REPEAT_2(x) x, x
#define REPEAT_4(x) REPEAT_2(x), REPEAT_2(x)
#define REPEAT_5(x) REPEAT_4(x), x
#define REPEAT_6(x) REPEAT_4(x), REPEAT_2(x)
#define REPEAT_7(x) REPEAT_4(x), REPEAT_2(x), x
#define REPEAT_8(x) REPEAT_4(x), REPEAT_4(x)
#define REPEAT_16(x) REPEAT_8(x), REPEAT_8(x)
#define SHOW_ARR(x) x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

inline void hmac512_ccode_msg37(__private const ulong c[4], __private const ulong M0, __private const ulong M1, __private const ulong M2,
                                ulong M3, ulong M4top5, ulong Hout[8]) {
  ulong inner[32], outer[32];
  inner[0] = c[0] ^ IPAD;
  inner[1] = c[1] ^ IPAD;
  inner[2] = c[2] ^ IPAD;
  inner[3] = c[3] ^ IPAD;
  outer[0] = c[0] ^ OPAD;
  outer[1] = c[1] ^ OPAD;
  outer[2] = c[2] ^ OPAD;
  outer[3] = c[3] ^ OPAD;
  #pragma unroll
  for (int i = 4; i < 16; i++) {
    inner[i] = IPAD;
    outer[i] = OPAD;
  }

  // inner: (K^ipad)||msg(37)||0x80||zeros||len(1320)
  inner[16] = M0;
  inner[17] = M1;
  inner[18] = M2;
  inner[19] = M3;
  inner[20] = M4top5 | ((ulong)0x80UL << 16);
  #pragma unroll
  for (int i = 21; i < 30; i++)
    inner[i] = 0;
  inner[30] = 0;
  inner[31] = (ulong)1320;
  sha512_hash_two_blocks_message(inner, Hout);
  outer[16] = Hout[0];
  outer[17] = Hout[1];
  outer[18] = Hout[2];
  outer[19] = Hout[3];
  outer[20] = Hout[4];
  outer[21] = Hout[5];
  outer[22] = Hout[6];
  outer[23] = Hout[7];
  outer[24] = 0x8000000000000000UL;
  #pragma unroll
  for (int i = 25; i < 30; i++)
    outer[i] = 0;
  outer[30] = 0;
  outer[31] = (ulong)1536;
  sha512_hash_two_blocks_message(outer, Hout);
}




