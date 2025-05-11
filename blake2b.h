#ifndef BLAKE2B_H
#define BLAKE2B_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLAKE2B_OUTBYTES 64

typedef struct {
  uint64_t h[8];
  uint64_t t[2];
  uint64_t f[2];
  uint8_t buf[2 * 128];
  size_t buflen;
  uint8_t last_node;
} blake2b_state;

int blake2b_init(blake2b_state *S, size_t outlen);
int blake2b_update(blake2b_state *S, const void *in, size_t inlen);
int blake2b_final(blake2b_state *S, void *out, size_t outlen);
int blake2b(void *out, size_t outlen, const void *in, size_t inlen, const void *key, size_t keylen);

#ifdef __cplusplus
}
#endif

#endif


