"""Relink the capture-only Vulkan worker from a matching non-CUDA KoboldCpp checkout."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess

REL = 'otherarch/sdcpp/src/core/ggml_runner'
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'compile-log', 'make', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = args.core.resolve()
    source = (core / (REL + '.cpp')).read_text(encoding='utf-8')
    include = Path(__file__).with_name('native_capture.inc').read_text(encoding='utf-8')
    if source.count(PATTERN) != 1 or source.count('using namespace sd;\n') != 1:
        raise RuntimeError('Native callback location changed. Use a supported core revision.')
    source = source.replace('using namespace sd;\n', 'using namespace sd;\n' + include, 1).replace(PATTERN, REPLACE, 1)
    cpp, obj, library = (out / name for name in ('ggml_runner.cpp', 'ggml_runner.o', 'koboldcpp_vulkan.dll'))
    cpp.write_text(source, encoding='utf-8')
    log = args.compile_log.read_text(encoding='utf-8', errors='replace').splitlines()
    compile_line = next(line for line in log if f' -c {REL}.cpp ' in line)
    link_line = next(line for line in log if ' -o koboldcpp_vulkan.dll ' in line)
    env = dict(os.environ, MAKE=args.make.resolve().as_posix())
    env['PATH'] = os.pathsep.join((str(args.make.resolve().parent),
                                  str(Path(shlex.split(compile_line)[0]).parent), env['PATH']))
    for index, (line, changes) in enumerate(((compile_line, {REL+'.cpp':cpp.as_posix(),REL+'.o':obj.as_posix()}),
                                             (link_line, {REL+'.o':obj.as_posix(),'koboldcpp_vulkan.dll':library.as_posix()}))):
        command = [changes.get(flag, flag) for flag in shlex.split(line)]
        if any('-fprofile-' in flag or 'llama_cuda' in flag or 'cublas' in flag for flag in command):
            raise RuntimeError('Expected the unprofiled Vulkan build recipe')
        if index == 0:
            command.append(f'-ffile-prefix-map={cpp.as_posix()}={REL}.cpp')
        subprocess.run(command, cwd=core, env=env, check=True)
    if out.as_posix().encode() in library.read_bytes():
        raise RuntimeError('Capture worker contains a private build path.')
    shutil.copyfile(core / 'koboldcpp.py', out / 'koboldcpp.py')
    with library.open('rb') as f:
        print(hashlib.file_digest(f,'sha256').hexdigest(), flush=True)


if __name__ == '__main__':
    main()
