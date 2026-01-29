


#define COPY_EIGHT(a, b)                                                       \
  (a)[0] = (b)[0], (a)[1] = (b)[1], (a)[2] = (b)[2], (a)[3] = (b)[3],          \
  (a)[4] = (b)[4], (a)[5] = (b)[5], (a)[6] = (b)[6], (a)[7] = (b)[7];


#define COPY_FOUR(a, b)                                                        \
  a[0] = b[0];                                                                 \
  a[1] = b[1];                                                                 \
  a[2] = b[2];                                                                 \
  a[3] = b[3];

#define COPY_FOUR_OFFSET(a, b, c)                                              \
  a[0] = b[0 + c];                                                             \
  a[1] = b[1 + c];                                                             \
  a[2] = b[2 + c];                                                             \
  a[3] = b[3 + c];

#define COPY_EIGHT_XOR(a, b)                                                   \
  (a)[0] ^= (b)[0];                                                            \
  (a)[1] ^= (b)[1];                                                            \
  (a)[2] ^= (b)[2];                                                            \
  (a)[3] ^= (b)[3];                                                            \
  (a)[4] ^= (b)[4];                                                            \
  (a)[5] ^= (b)[5];                                                            \
  (a)[6] ^= (b)[6];                                                            \
  (a)[7] ^= (b)[7];


#define DEBUG_ARRAY(name, array, len)                                          \
  do {                                                                         \
    for (uint i = 0; i < (len); i++) {                                         \
      printf("%s[%d] = 0x%016lxUL\n",name,i, (array)[i]);                                       \
    }                                                                          \
                                                                               \
  } while (0)

