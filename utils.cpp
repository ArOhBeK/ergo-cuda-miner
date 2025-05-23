#include "utils.h"

std::vector<uint8_t> decimal_to_target_bytes(const std::string& decimal) {
    mpz_t target_int;
    mpz_init(target_int);
    mpz_set_str(target_int, decimal.c_str(), 10);

    uint8_t buf[32] = {0};
    size_t count = 0;
    mpz_export(buf, &count, 1, 1, 1, 0, target_int);

    std::vector<uint8_t> out(32, 0);
    if (count > 0) {
        memcpy(out.data() + (32 - count), buf, count);
    }

    mpz_clear(target_int);
    return out;
}
