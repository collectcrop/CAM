#include <cstddef>
#include <cstdint>
namespace wiki_ts_rmi_linear_spline_linear_131072 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3145744;
const uint64_t BUILD_TIME_NS = 4858601727;
const char NAME[] = "wiki_ts_rmi_linear_spline_linear_131072";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
