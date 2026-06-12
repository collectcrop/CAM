#include <cstddef>
#include <cstdint>
namespace fb_rmi_linear_spline_linear_512 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 12304;
const uint64_t BUILD_TIME_NS = 11603409708;
const char NAME[] = "fb_rmi_linear_spline_linear_512";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
