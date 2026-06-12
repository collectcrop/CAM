#include <cstddef>
#include <cstdint>
namespace fb_rmi_linear_spline_linear_128 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3088;
const uint64_t BUILD_TIME_NS = 11106007023;
const char NAME[] = "fb_rmi_linear_spline_linear_128";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
