"""Build the single-file Windows release from pinned public dependencies."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
APP = HERE / 'app'
SYSTEM = Path(os.environ['SystemRoot'])
CRT = ('concrt140.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
       'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'vcomp140.dll',
       'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def acquire(cache, spec):
    filename = spec['name']
    path = cache / filename
    if not path.is_file():
        print('Downloading', filename, flush=True)
        with tempfile.TemporaryDirectory(prefix='download-', dir=cache) as temporary:
            part = Path(temporary) / filename
            with urllib.request.urlopen(spec['url'], timeout=90) as response, part.open('xb') as output:
                shutil.copyfileobj(response, output)
            if digest(part) != spec['sha256']:
                raise ValueError(f'Unexpected archive digest: {filename}')
            part.rename(path)
    if digest(path) != spec['sha256']:
        raise ValueError(f'Unexpected archive digest: {filename}')
    return path


def expand_cab(source, target, pattern='*'):
    target.mkdir()
    subprocess.run([str(SYSTEM / 'System32' / 'expand.exe'), '-F:' + pattern, str(source), str(target)],
                   check=True, stdout=subprocess.DEVNULL)


def unpack_crt(installer, workspace):
    # WixAttachedContainer for the exact, SHA-pinned 14.51.36247 x64 installer.
    data = installer.read_bytes()
    offset = 630000
    if data[offset:offset + 4] != b'MSCF':
        raise ValueError('Unexpected Microsoft installer layout.')
    length, = struct.unpack_from('<I', data, offset + 8)
    if not 36 < length <= len(data) - offset:
        raise ValueError('Invalid Microsoft cabinet length.')
    cabinet = workspace / 'attached.cab'
    cabinet.write_bytes(data[offset:offset + length])
    nested = workspace / 'cabinet'
    expand_cab(cabinet, nested, 'a4')
    x64 = workspace / 'crt-x64'
    expand_cab(nested / 'a4', x64)
    return {name: x64 / (name + '_amd64') for name in CRT}


def unpack_native(archive, destination, executable):
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = PurePosixPath(entry.filename).name
            if entry.is_dir():
                continue
            if name.lower().endswith('-impl.dll') and name.lower() != 'llama-quantize-impl.dll':
                continue
            if name.lower() == executable or name.lower().endswith('.dll') or 'license' in name.lower():
                with (destination / name).open('xb') as out, source.open(entry) as stream:
                    shutil.copyfileobj(stream, out)
    if not (destination / executable).is_file():
        raise ValueError(f'Missing executable {executable}.')


def copy_crt(crt, destination):
    for name, source in crt.items():
        shutil.copyfile(source, destination / name)


def included(relative):
    if relative.suffix.lower() in {'.pdb', '.pyc', '.lib', '.obj', '.whl', '.gguf', '.safetensors', '.onnx'}:
        return False
    if relative.name in {'direct_url.json', 'pyvenv.cfg'} or '__pycache__' in relative.parts:
        return False
    # The interpreter and its packages are bundled as files. Build headers and pip stay in staging.
    if relative.parts[0] == 'python':
        if relative.parts[1] in {'include', 'libs', 'Scripts'}:
            return False
        if len(relative.parts) > 3 and relative.parts[1:3] == ('Lib', 'site-packages'):
            package = relative.parts[3]
            if package == 'pip' or package.startswith('pip-'):
                return False
            if relative.parts[3:5] in {('torch', 'include'), ('torch', 'test')}:
                return False
            if relative.parts[3:] == ('transformers', 'testing_utils.py'):
                return False
        if 'tests' in relative.parts or relative.parts[1:3] == ('Lib', 'test'):
            return False
    return True


def build_vulkan(args, destination):
    destination.mkdir()
    source = APP / 'vulkan'
    for mode in ('gsq', 'rco', 'rco_full', 'loss', 'actcal', 'actcal_k', 'macaq_flip'):
        subprocess.run([str(args.glslc), '--target-env=vulkan1.1', '-O',
                        str(source / f'{mode}.comp'), '-o', str(destination / f'{mode}.spv')], check=True)
    subprocess.run([str(args.cxx), '-std=c++17', '-O2', '-municode', '-static', '-static-libgcc', '-static-libstdc++',
                    '-I' + str(args.vulkan_include), str(source / 'vulkan-gsq.cpp'),
                    str(args.vulkan_import), '-o', str(destination / 'vulkan-gsq.exe')], check=True)


def build(args):
    pins = json.loads((HERE / 'dependencies.lock.json').read_text(encoding='utf-8'))
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = acquire(cache, pins['python'])
    installer = acquire(cache, pins['crt'])
    llama = acquire(cache, pins['llama'])
    sd = acquire(cache, pins['sd'])
    wheels = [acquire(cache, pin) for pin in pins['wheels']]
    with tempfile.TemporaryDirectory(prefix='gguf-build-', dir=output.parent) as work:
        workspace = Path(work)
        root = workspace / 'payload'
        root.mkdir()
        with tarfile.open(runtime) as archive:
            archive.extractall(root, filter='data')
        python = root / 'python'
        crt = unpack_crt(installer, workspace)
        copy_crt(crt, python)
        subprocess.run([str(python / 'python.exe'), '-I', '-B', '-m', 'pip', 'install', '--no-index',
                        '--no-deps', '--no-compile', '--target', str(python / 'Lib' / 'site-packages'),
                        *map(str, wheels)], check=True)
        app = root / 'app'
        backends = app / 'backends'
        unpack_native(llama, backends / 'llama', 'llama-quantize.exe')
        unpack_native(sd, backends / 'sd', 'sd-cli.exe')
        for engine in ('llama', 'sd'):
            copy_crt(crt, backends / engine)
        shutil.copytree(args.hf, backends / 'hf', ignore=shutil.ignore_patterns('__pycache__', 'tests', 'examples'))
        for source in APP.glob('*.py'):
            shutil.copyfile(source, app / source.name)
        assets = APP / 'assets'
        shutil.copytree(assets, app / assets.name)
        for filename in ('LICENSE.md', 'NOTICE.md', 'dependencies.lock.json'):
            shutil.copyfile(HERE / filename, app / filename)
        shutil.copytree(HERE / 'licenses', app / 'licenses')
        if args.vulkan_include:
            build_vulkan(args, app / 'vulkan')
        if args.native_capture:
            directory = app / 'capture-runtime'
            directory.mkdir()
            for rel in (Path('koboldcpp.py'), Path('koboldcpp_vulkan.dll'), Path('embd_res/yue2_qwen_tiktoken.embd')):
                (directory / rel.parent).mkdir(parents=True,exist_ok=True)
                shutil.copyfile(args.native_capture / rel, directory / rel)

        payload = workspace / 'converter.zip'
        with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(root.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if included(relative):
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f'File leaves payload: {relative}')
                    archive.write(path, relative.as_posix())
        with zipfile.ZipFile(payload) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f'Corrupt payload member: {corrupt}')

        compiler = SYSTEM / 'Microsoft.NET' / 'Framework64' / 'v4.0.30319' / 'csc.exe'
        candidate = workspace / 'GGUF-Converter.exe'
        subprocess.run([str(compiler), '/nologo', '/platform:x64', '/optimize+', '/target:winexe',
                        '/r:System.Windows.Forms.dll', '/r:System.IO.Compression.dll',
                        f'/win32icon:{assets / "gguf.ico"}',
                        f'/resource:{payload},converter.zip', f'/out:{candidate}', str(HERE / 'Launcher.cs')], check=True)
        candidate.rename(output)
    print(output.name, output.stat().st_size, 'bytes', digest(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=HERE / 'build-assets')
    parser.add_argument('--hf', type=Path, default=HERE / 'vendor' / 'hf')
    parser.add_argument('--output', type=Path, default=HERE / 'dist' / 'GGUF-Converter.exe')
    parser.add_argument('--cxx', type=Path)
    parser.add_argument('--glslc', type=Path)
    parser.add_argument('--vulkan-include', type=Path)
    parser.add_argument('--vulkan-import', type=Path)
    parser.add_argument('--native-capture', type=Path)
    args = parser.parse_args()
    options = (args.cxx, args.glslc, args.vulkan_include, args.vulkan_import)
    if any(options) and not all(options):
        parser.error('Vulkan builds require --cxx, --glslc, --vulkan-include and --vulkan-import together.')
    if args.native_capture and not all(options):
        parser.error('The native capture worker requires a Vulkan build.')
    build(args)
