#include <cstddef>
#include <cstdint>
namespace osm_cellids_rmi_linear_spline_linear_4096 {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 98320;
const uint64_t BUILD_TIME_NS = 12306166503;
const char NAME[] = "osm_cellids_rmi_linear_spline_linear_4096";
uint64_t lookup_with_leaf(uint64_t key, size_t* err, size_t* leaf);
uint64_t lookup(uint64_t key, size_t* err);
}
