#include <cstddef>
#include <cstdint>
namespace fb_rmi_linear_spline_linear_8192 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 196624;
const uint64_t BUILD_TIME_NS = 10109903818;
const char NAME[] = "fb_rmi_linear_spline_linear_8192";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
