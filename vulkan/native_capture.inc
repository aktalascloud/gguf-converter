#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

static bool gsq_capture_activation(ggml_tensor * node, bool ask, void * directory) {
    if (node->op != GGML_OP_MUL_MAT || !node->src[0] || !node->src[1]) return !ask;
    const auto * weight = node->src[0];
    const auto * input = node->src[1];
    const bool f32 = input->type == GGML_TYPE_F32;
    const bool f16 = input->type == GGML_TYPE_F16;
    if (weight->view_src != nullptr || (!f32 && !f16) || input->nb[0] != (f32 ? sizeof(float) : sizeof(ggml_fp16_t)) ||
        input->ne[0] != weight->ne[0] || input->ne[0] < 32 || input->ne[0] > 65536 ||
        input->ne[0] % 32 || ggml_nrows(input) < 2) return !ask;

    const auto root = std::filesystem::u8path(static_cast<const char *>(directory));
    static const std::vector<std::string> names = [&root] {
        std::ifstream plan(root / "tensors.txt");
        std::vector<std::string> lines;
        for (std::string name; lines.size() < 4096 && std::getline(plan, name);) {
            if (!name.empty() && name.back() == '\r') name.pop_back();
            if (!name.empty()) lines.push_back(name);
        }
        return lines;
    }();
    static std::set<size_t> captured;
    const std::string actual = weight->name;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto & name = names[i];
        const size_t end = actual.size() >= name.size() ? actual.size() - name.size() : std::string::npos;
        if (end == std::string::npos || (end && actual[end - 1] != '.') ||
            actual.compare(end, name.size(), name) || captured.count(i)) continue;
        if (ask) return true;
        const uint32_t count = static_cast<uint32_t>(std::min<int64_t>(32, ggml_nrows(input)));
        const uint32_t width = static_cast<uint32_t>(input->ne[0]);
        std::vector<float> values(static_cast<size_t>(count) * width);
        std::vector<ggml_fp16_t> half(f16 ? width : 0);
        for (uint32_t row = 0; row < count; ++row) {
            const size_t from = static_cast<size_t>(row * (ggml_nrows(input) - 1) / (count - 1));
            const size_t in_plane = from / input->ne[1];
            const size_t offset = (from % input->ne[1]) * input->nb[1] + (in_plane % input->ne[2]) * input->nb[2]
                                  + (in_plane / input->ne[2]) * input->nb[3];
            auto * dst = values.data() + static_cast<size_t>(row) * width;
            if (f32) ggml_backend_tensor_get(input, dst, offset, width * sizeof(float));
            else {
                ggml_backend_tensor_get(input, half.data(), offset, width * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(half.data(), dst, width);
            }
        }
        std::ofstream stream(root / (std::to_string(i) + ".f32"), std::ios::binary);
        const uint32_t shape[] = {count, width};
        stream.write(reinterpret_cast<const char *>(shape), sizeof(shape));
        stream.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        stream.close();
        if (stream) captured.insert(i);
        break;
    }
    return !ask;
}
