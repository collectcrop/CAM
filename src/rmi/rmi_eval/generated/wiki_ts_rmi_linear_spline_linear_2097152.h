#include <cstddef>
#include <cstdint>
namespace wiki_ts_rmi_linear_spline_linear_2097152 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 50331664;
const uint64_t BUILD_TIME_NS = 4961729966;
const char NAME[] = "wiki_ts_rmi_linear_spline_linear_2097152";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
