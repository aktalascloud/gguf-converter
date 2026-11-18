"""Relink the capture-only Vulkan worker from a matching non-CUDA KoboldCpp checkout."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess

REL = 'otherarch/sdcpp/src/core/ggml_runner'
AUDIO = 'otherarch/audiocpp/upstream/src/framework/core/backend'
TENSORS = 'otherarch/audiocpp/upstream/src/framework/assets/tensor_source'
PATTERN = '    if (scheduler != nullptr) {\n        if (sd_get_backend_eval_callback()'
REPLACE = '''    const char * capture = std::getenv("KCPP_GGUF_ACTIVATIONS");
    if (capture && *capture) {
        if (scheduler != nullptr) {
            ggml_backend_sched_set_eval_callback(scheduler, gsq_capture_activation, const_cast<char *>(capture));
            status = ggml_backend_sched_graph_compute(scheduler, graph);
            ggml_backend_sched_set_eval_callback(scheduler, nullptr, nullptr);
        } else {
            status = sd_backend_graph_compute_with_eval_callback(runtime_backend, graph, gsq_capture_activation,
                                                               const_cast<char *>(capture));
        }
    } else if (scheduler != nullptr) {
        if (sd_get_backend_eval_callback()'''
AUDIO_REPLACE = '''    if (const char * directory = std::getenv("KCPP_GGUF_ACTIVATIONS");
        directory && std::getenv("KCPP_GGUF_CAPTURE_MUSIC") && !plan && is_vulkan_backend_handle(backend)) {
        return sd_backend_graph_compute_with_eval_callback(backend, graph, gsq_capture_activation,
                                                           const_cast<char *>(directory));
    }
    return plan != nullptr
        ? ggml_backend_graph_plan_compute(backend, plan)'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'compile-log', 'make', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = args.core.resolve()
    include = Path(__file__).with_name('native_capture.inc').read_text(encoding='utf-8')
    source = (core / (REL + '.cpp')).read_text(encoding='utf-8')
    if source.count(PATTERN) != 1 or source.count('using namespace sd;\n') != 1:
        raise RuntimeError('Native callback location changed. Use a supported core revision.')
    source = source.replace('using namespace sd;\n', 'using namespace sd;\n' + include, 1).replace(PATTERN, REPLACE, 1)
    audio = (core / (AUDIO + '.cpp')).read_text(encoding='utf-8')
    marker = 'namespace engine::core {\n'
    dispatch = '    return plan != nullptr\n        ? ggml_backend_graph_plan_compute(backend, plan)'
    if audio.count(marker) != 1 or audio.count(dispatch) != 1:
        raise RuntimeError('Audio backend callback location changed. Use a supported core revision.')
    declaration = ('ggml_status sd_backend_graph_compute_with_eval_callback(ggml_backend_t, ggml_cgraph *,\n'
                   '    bool (*)(ggml_tensor *, bool, void *), void *);\n')
    audio = audio.replace(marker, include + '\n' + declaration + '\n' + marker, 1).replace(dispatch, AUDIO_REPLACE, 1)
    tensors = (core / (TENSORS + '.cpp')).read_text(encoding='utf-8')
    load = ('const std::vector<int64_t> & expected_shape) const override {\n'
            '        const auto & info = require_info(name);\n'
            '        validate_expected_shape(name, info.shape, expected_shape);')
    if tensors.count(load) != 1:
        raise RuntimeError('Audio GGUF upload location changed. Use a supported core revision.')
    tensors = '#include <cstdlib>\n' + tensors.replace(load, load + '''
        if (tensor && std::getenv("KCPP_GGUF_CAPTURE_MUSIC")) {
            ggml_set_name(tensor, info.logical_name.c_str());
        }''', 1)
    library = out / 'koboldcpp_vulkan.dll'
    log = args.compile_log.read_text(encoding='utf-8', errors='replace').splitlines()
    link_line = next(line for line in log if ' -o koboldcpp_vulkan.dll ' in line)
    compile_line = next(line for line in log if f' -c {REL}.cpp ' in line)
    env = dict(os.environ, MAKE=args.make.resolve().as_posix())
    env['PATH'] = os.pathsep.join((str(args.make.resolve().parent),
                                  str(Path(shlex.split(compile_line)[0]).parent), env['PATH']))
    def run(line, changes, extra=()):
        command = [changes.get(flag, flag) for flag in shlex.split(line)]
        if any('-fprofile-' in flag or 'llama_cuda' in flag or 'cublas' in flag for flag in command):
            raise RuntimeError('Expected the unprofiled Vulkan build recipe')
        subprocess.run([*command,*extra], cwd=core, env=env, check=True)

    objects = {'koboldcpp_vulkan.dll':library.as_posix()}
    for rel, body in ((REL, source),(AUDIO, audio),(TENSORS, tensors)):
        stem = Path(rel).name
        cpp = out / (stem + '.cpp')
        obj = out / (stem + '.o')
        cpp.write_text(body, encoding='utf-8')
        line = next(item for item in log if f' -c {rel}.cpp ' in item)
        objects[rel+'.o'] = obj.as_posix()
        run(line, {rel+'.cpp':cpp.as_posix(), rel+'.o':obj.as_posix()},
            [f'-ffile-prefix-map={cpp.as_posix()}={rel}.cpp'])
    run(link_line, objects)
    if out.as_posix().encode() in library.read_bytes():
        raise RuntimeError('Capture worker contains a private build path.')
    shutil.copyfile(core / 'koboldcpp.py', out / 'koboldcpp.py')
    tokenizer = Path('embd_res/yue2_qwen_tiktoken.embd')
    (out / tokenizer.parent).mkdir()
    shutil.copyfile(core / tokenizer, out / tokenizer)
    with library.open('rb') as f:
        print(hashlib.file_digest(f,'sha256').hexdigest(), flush=True)


if __name__ == '__main__':
    main()
