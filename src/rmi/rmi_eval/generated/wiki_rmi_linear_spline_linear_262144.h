#include <cstddef>
#include <cstdint>
namespace wiki_rmi_linear_spline_linear_262144 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 6291472;
const uint64_t BUILD_TIME_NS = 12308250453;
const char NAME[] = "wiki_rmi_linear_spline_linear_262144";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
