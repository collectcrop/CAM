#include <cstddef>
#include <cstdint>
namespace wiki_ts_rmi_linear_spline_linear_1024 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 24592;
const uint64_t BUILD_TIME_NS = 4089933385;
const char NAME[] = "wiki_ts_rmi_linear_spline_linear_1024";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
