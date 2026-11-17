"""Activation-guided GSQ and bounded RCO for standard K-quants."""
from contextlib import ExitStack
from itertools import product
from pathlib import Path
import tempfile

import numpy as np


BLOCK = 256
BITS = (2, 3, 4)
BLOCK_BYTES = {2: 84, 3: 110, 4: 144, 6: 210, 8: 272}


def pack_k(codes, scales, bits, *, legacy_q4=False):
    """Pack one signed GSQ grid and eight scales per 256 weights."""
    count = len(codes)
    if not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError('Invalid GSQ group scale.')
    zero_point = 1 << (bits - 1)
    if (codes < -zero_point).any() or (codes >= zero_point).any():
        raise ValueError('GSQ code exceeds the selected grid.')
    if bits == 8 or legacy_q4:
        d = scales.astype('<f2')
        if not np.isfinite(d).all():
            raise ValueError('GSQ scale exceeds GGUF FP16 range.')
        if legacy_q4:
            pairs = (codes + 8).astype(np.uint8).reshape(count, 8, 2, 16)
            payload = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
        else:
            payload = codes.astype(np.int8).view(np.uint8).reshape(count, 8, 32)
        return np.concatenate((d.view(np.uint8).reshape(count, 8, 2), payload), axis=2).reshape(count, -1)
    limit = {2: 15, 3: 31, 4: 63, 6: 127}[bits]
    d = (scales.max(axis=1, keepdims=True) / limit).astype('<f2')
    if not np.isfinite(d).all() or (d.astype(np.float32) * (1 << (bits - 1)) > np.finfo(np.float16).max).any():
        raise ValueError('GSQ scale exceeds GGUF FP16 range.')
    encoded = np.divide(scales, d, where=d > 0, out=np.zeros_like(scales))
    s = np.clip(np.rint(encoded), 0, limit).astype(np.uint8)
    q = (codes + zero_point).astype(np.uint8)
    if bits == 6:
        low = (q & 15).reshape(count, 2, 2, 64)
        low = (low[:, :, 0] | (low[:, :, 1] << 4)).reshape(count, 128)
        high = ((q >> 4).reshape(count, 2, 4, 32) << np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None])
        high = np.bitwise_or.reduce(high, axis=2).reshape(count, 64)
        return np.concatenate((low, high, np.repeat(s, 2, axis=1), d.view(np.uint8)), axis=1)
    if bits == 4:
        a, b = np.hsplit(s, 2)
        scales_min = np.concatenate((a | ((b & 48) << 2), a | ((b & 48) << 2),
                                     (b & 15) | ((b & 15) << 4)), axis=1)
        nibble = q.reshape(count, 4, 2, 32)
        packed = (nibble[:, :, 0] | (nibble[:, :, 1] << 4)).reshape(count, 128)
        return np.concatenate((d.view(np.uint8), (d.astype(np.float32) * 8).astype('<f2').view(np.uint8),
                               scales_min, packed), axis=1)
    low_q = ((q & 3).reshape(count, 2, 4, 32) << np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None])
    low_q = np.bitwise_or.reduce(low_q, axis=2).reshape(count, 64)
    s = np.repeat(s, 2, axis=1)
    if bits == 2:
        return np.concatenate((s | (s << 4), low_q, d.view(np.uint8),
                               (d.astype(np.float32) * 2).astype('<f2').view(np.uint8)), axis=1)
    hmask = ((q >> 2).reshape(count, 8, 32) << np.arange(8, dtype=np.uint8)[None, :, None])
    hmask = np.bitwise_or.reduce(hmask, axis=1)
    s += 32
    a, b = np.hsplit(s, 2)
    lower = (a & 15) | ((b & 15) << 4)
    upper = ((s >> 4).reshape(count, 4, 4) << np.array([0, 2, 4, 6], dtype=np.uint8)[None, :, None])
    upper = np.bitwise_or.reduce(upper, axis=1)
    return np.concatenate((hmask, low_q, lower, upper, d.view(np.uint8)), axis=1)


def dispatch(run, binaries, work, mode, payload, count, steps, samples, bits, seed, budget=0, repeats=1):
    data, result = work / 'input.bin', work / 'output.bin'
    np.asarray(payload, dtype='<f4').tofile(data)
    shader = 'gsq' if mode == 'gsq_shared' else mode
    run(binaries / 'vulkan-gsq.exe', mode, data, result, binaries / f'{shader}.spv',
        count, steps, samples, bits, seed, budget, repeats)
    output = np.fromfile(result, dtype='<f4')
    expected = {'gsq': count * 266, 'rco': count * (bits + 1) + 4, 'loss': count}[shader.removesuffix('_full')]
    if output.size != expected or not np.isfinite(output).all():
        raise RuntimeError('Invalid Vulkan calibration result.')
    return output


def rtn(w, bits):
    low = -(1 << (bits - 1))
    high = -low - 1
    groups = w.reshape(-1, 8, 32)
    scale = np.maximum(np.maximum(groups.max(axis=-1) / high, groups.min(axis=-1) / low), 1e-7)
    return np.clip(np.rint(w / np.repeat(scale, 32, axis=1)), low, high), scale


def learn(run, binaries, work, weights, activation, bits, steps, seed, dequantize, *, legacy_q4=False):
    n = len(weights)
    packed = np.concatenate((weights.ravel(), activation.ravel()))
    raw = dispatch(run, binaries, work, 'gsq', packed, n, steps, activation.shape[1], bits, seed)
    codes = raw[:n * BLOCK].reshape(n, BLOCK)
    scales = raw[n * 258:].reshape(n, 8)
    learned = pack_k(codes, scales, bits, legacy_q4=legacy_q4)
    baseline = pack_k(*rtn(weights, bits), bits, legacy_q4=legacy_q4)
    # Compare both candidates in the format actually used by the inference backend.
    def loss(q):
        residual = activation @ (dequantize(q) - weights)[..., None]
        return np.mean(residual[..., 0] ** 2, axis=1)
    chosen = np.where((loss(learned) <= loss(baseline))[:, None], learned, baseline)
    return chosen, baseline


def build_variants(converter, binaries, work, tensor, inputs, steps, seed, quants, bits_options, types, maps,
                   model_loss=False):
    columns, rows = map(int, tensor.shape)
    blocks = columns // BLOCK
    if not 1 <= blocks <= 4096:
        raise ValueError(f'Unsupported GSQ width for {tensor.name}.')
    x = np.load(inputs, allow_pickle=False)
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != columns or not 2 <= len(x) <= 32:
        if hasattr(x, 'close'):
            x.close()
        raise ValueError(f'{tensor.name} needs a .npy array with 2–32 activation rows of width {columns}.')
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'Non-finite activation rows for {tensor.name}.')
    bx = x.reshape(len(x), blocks, BLOCK).transpose(1, 0, 2)
    # Keep dispatch duration bounded on display GPUs with the Windows watchdog.
    block_limit = min(4096, (4096 * 24 * 16) // (steps * len(x)))
    batch_rows = max(1, block_limit // blocks)
    arrays = []
    rounding = []
    normalized = []
    for bits, kind in zip(bits_options, types):
        converter.log(f'Vulkan GSQ · {tensor.name} · {kind.name}')
        progress_label = f'Калибровка GSQ · {kind.name}'
        converter.stage(progress_label, 0, rows)
        block_bytes = BLOCK_BYTES[bits]
        candidate = np.memmap(work / f'q{bits}.bin', dtype=np.uint8, mode='w+',
                              shape=(rows, blocks * block_bytes))
        maps.callback(candidate._mmap.close)
        baseline = None
        if model_loss:
            baseline = np.memmap(work / f'q{bits}-rtn.bin', dtype=np.uint8, mode='w+', shape=candidate.shape)
            maps.callback(baseline._mmap.close)
        error, energy = 0., 0.
        def decode(value):
            return quants.dequantize(value, kind)
        for start in range(0, rows, batch_rows):
            original = quants.dequantize(tensor.data[start:start + batch_rows], tensor.tensor_type).astype(np.float32)
            weights = original.reshape(-1, BLOCK)
            activation = np.tile(bx, (len(original), 1, 1))
            packed, rounded = learn(converter.run, binaries, work, weights, activation,
                                    bits, steps, (seed + start) % (1 << 32), decode)
            candidate[start:start + len(original)] = packed.reshape(len(original), -1)
            if baseline is not None:
                baseline[start:start + len(original)] = rounded.reshape(len(original), -1)
            converter.stage(progress_label, start + len(original), rows)
            dense = decode(packed).reshape(original.shape)
            delta = (dense - original) @ x.T
            score = original @ x.T
            error += float(np.sum(delta ** 2, dtype=np.float64))
            energy += float(np.sum(score ** 2, dtype=np.float64))
        candidate.flush()
        arrays.append(candidate)
        if baseline is not None:
            baseline.flush()
            rounding.append(baseline)
        normalized.append(error / max(energy, 1e-30))
    return arrays, normalized, rounding


def optimize_same_layout(converter, manifest, base, reference, target, output, gguf_module, *,
                         capture=None, detect=None):
    """Calibrate selected quantized tensors against matching higher-precision weights."""
    tensors = target.get('tensors')
    steps = target.get('steps', 24)
    if not isinstance(tensors, dict) or not 1 <= len(tensors) <= 8 or type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError('Укажите 1–8 тензоров и от 1 до 128 шагов калибровки.')
    if base.endianess.name != 'LITTLE' or reference.endianess.name != 'LITTLE':
        raise ValueError('Vulkan-калибровка принимает little-endian GGUF.')
    base_arch, ref_arch = (r.get_field('general.architecture') for r in (base, reference))
    if base_arch is None or ref_arch is None or base_arch.contents() != ref_arch.contents():
        raise ValueError('Архитектура исходных весов не совпадает с целевым GGUF.')
    base_weights = {t.name: t for t in base.tensors}
    refs = {t.name: t for t in reference.tensors}
    layouts = {'Q2_K':2, 'Q3_K':3, 'Q4_K':4, 'Q4_0':4, 'Q6_K':6, 'Q8_0':8}
    entries = []
    for name, relative in tensors.items():
        t, ref = base_weights.get(name), refs.get(name)
        if t is None or ref is None or len(t.shape) != 2 or int(t.shape[0]) % BLOCK or tuple(t.shape) != tuple(ref.shape):
            raise ValueError(f'{name}: у исходного и целевого веса должны совпадать двумерные формы с шириной, кратной 256.')
        if t.tensor_type.name not in layouts or ref.tensor_type.name not in ('F32', 'F16', 'BF16'):
            raise ValueError(f'{name}: нужны исходный F32/F16/BF16 и поддерживаемый квант GGUF.')
        if relative is None and capture is None:
            raise ValueError(f'{name}: укажите активации .npy или настройки capture.')
        if relative is not None and (not isinstance(relative, str) or Path(relative).suffix.lower() != '.npy'):
            raise ValueError(f'{name}: ожидаются активации .npy.')
        entries.append((t, ref, relative))
    binaries = Path(__file__).resolve().parent / 'vulkan'
    dequant = gguf_module('quants').dequantize
    with tempfile.TemporaryDirectory(prefix='.gguf-calibrate-', dir=output.parent) as scratch, ExitStack() as files:
        root=Path(scratch)
        automatic=[t.name for t,_,relative in entries if relative is None]
        recorded={}
        if automatic:
            from native_capture import collect
            folder=root/'capture'
            folder.mkdir()
            converter.stage('Сбор активаций Vulkan')
            recorded,_,_=collect(converter,manifest,Path(base.data.filename).resolve(strict=True),
                                 capture,automatic,folder,detect)
        replacements={}
        for i,(t,ref,relative) in enumerate(entries):
            work=root/str(i)
            work.mkdir()
            data = recorded[t.name] if relative is None else (manifest.parent / relative).resolve(strict=True)
            x=np.load(data,allow_pickle=False)
            columns,rows=map(int,t.shape)
            if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[1]!=columns or not 8<=len(x)<=32 or not np.isfinite(x).all():
                if hasattr(x,'close'):
                    x.close()
                raise ValueError(f'{t.name}: нужны 8–32 конечных строки активаций шириной {columns}.')
            blocks=columns//BLOCK
            x=x.astype(np.float32,copy=False)
            training=x[:3*len(x)//4]
            validation=x[len(training):]
            train_blocks=training.reshape(len(training),blocks,BLOCK).transpose(1,0,2)
            batch=max(1,min(4096,4096*24*16//(steps*len(training)))//blocks)
            bits=layouts[t.tensor_type.name]
            typ=t.tensor_type
            out=np.memmap(work/'weights.bin',dtype=np.uint8,mode='w+',shape=t.data.shape)
            files.callback(out._mmap.close)
            before=after=0.
            for start in range(0, rows, batch):
                part=slice(start,min(rows,start+batch))
                expected=dequant(ref.data[part],ref.tensor_type).astype(np.float32)
                count=len(expected)
                weights=expected.reshape(-1,BLOCK)
                old=t.data[part].reshape(len(weights),-1)
                decode=lambda q:dequant(q,typ)
                probes=np.tile(train_blocks,(count,1,1))
                learnt,_=learn(converter.run,binaries,work,weights,probes,
                               bits,steps,1729+start,decode,legacy_q4=typ.name=='Q4_0')
                old_delta=(probes @ (decode(old)-weights)[...,None])[...,0]
                new_delta=(probes @ (decode(learnt)-weights)[...,None])[...,0]
                improved=(new_delta**2).sum(axis=1)<(old_delta**2).sum(axis=1)
                selected=np.where(improved[:,None],learnt,old)
                out[part]=selected.reshape(count,-1)
                old_full=(decode(old).reshape(count,columns)-expected) @ validation.T
                new_full=(decode(selected).reshape(count,columns)-expected) @ validation.T
                before+=float(np.sum(old_full**2,dtype=np.float64))
                after+=float(np.sum(new_full**2,dtype=np.float64))
                converter.stage(f'Калибровка {typ.name} · {i+1}/{len(entries)}',part.stop,rows)
            if after<before:
                replacements[t.name]=out,typ
            converter.log(f'{t.name} · ошибка на отложенных активациях {before:.8g} → {min(after,before):.8g} · {typ.name}')
        converter.log(f'Тензоров с улучшением по проверочным активациям: {len(replacements)}/{len(entries)}')
        return converter.write_mixed_gguf(base,output,replacements,preserve_file_type=True)


def optimize(converter, manifest, base, target, output, gguf_module, tier, capture=None, detect=None,
             model_loss=False):
    """Write a complete GGUF from bounded per-tensor search and external activation rows."""
    tensors = target.get('tensors')
    budget = target.get('max_tensor_bytes')
    steps = target.get('steps', 24)
    if not isinstance(tensors, dict) or not 1 <= len(tensors) <= 8 or type(budget) is not int or budget <= 0:
        raise ValueError('Specify 1–8 tensors and a positive max_tensor_bytes for Vulkan calibration.')
    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError('GSQ steps must be between 1 and 128.')
    if type(model_loss) is not bool or (model_loss and capture is None):
        raise ValueError('Для оценки целой модели укажите model_loss: true и настройки capture.')
    if base.endianess.name != 'LITTLE':
        raise ValueError('Vulkan calibration requires a little-endian GGUF.')
    binaries = Path(__file__).resolve().parent / 'vulkan'
    if not (binaries / 'vulkan-gsq.exe').is_file():
        raise FileNotFoundError('Vulkan helper is missing. See the build instructions.')
    quants = gguf_module('quants')
    qtypes = gguf_module('constants').GGMLQuantizationType
    bits_options = (*BITS, tier) if tier in (6, 8) else BITS
    types = tuple(qtypes['Q8_0' if b == 8 else f'Q{b}_K'] for b in bits_options)
    weights = {t.name: t for t in base.tensors}
    entries = []
    for name, path in tensors.items():
        t = weights.get(name)
        if t is None or len(t.shape) != 2 or int(t.shape[0]) % BLOCK:
            raise ValueError(f'{name} must be an existing two-dimensional GGUF weight with width divisible by 256.')
        if path is None and capture is not None:
            entries.append((t, None))
            continue
        if not isinstance(path, str) or Path(path).suffix.lower() != '.npy':
            raise ValueError(f'{name} requires a .npy activation array.')
        entries.append((t, (manifest.parent / path).resolve(strict=True)))
    retained = sum(t.n_bytes for t in base.tensors if t.name not in tensors)
    costs = [[t.n_elements // BLOCK * BLOCK_BYTES[b] for b in bits_options] for t, _ in entries]
    low, high = sum(row[0] for row in costs), sum(row[-1] for row in costs)
    available = budget - retained
    if available < low:
        raise ValueError(f'Tensor-data budget must be at least {retained + low} bytes.')
    with tempfile.TemporaryDirectory(prefix='.gguf-vulkan-', dir=output.parent) as scratch, ExitStack() as maps:
        root = Path(scratch)
        automatic = [t.name for t, path in entries if path is None]
        reference = Path(base.data.filename).resolve(strict=True) if automatic or model_loss else None
        if automatic or model_loss:
            from native_capture import collect
            directory = root / 'capture'
            directory.mkdir()
            converter.stage('Сбор активаций Vulkan')
            recorded, settings, payload = collect(converter, manifest, reference, capture, automatic, directory, detect)
            entries = [(t, path if path is not None else recorded[t.name]) for t, path in entries]
        variants = []
        rounded = []
        response = np.zeros((len(entries), len(bits_options), len(entries)), dtype=np.float32)
        for i, (t, path) in enumerate(entries):
            work = root / str(i)
            work.mkdir()
            arrays, errors, rounding = build_variants(converter, binaries, work, t, path, steps, 1729, quants,
                                                       bits_options, types, maps, model_loss)
            variants.append(arrays)
            rounded.append(rounding)
            # Orthogonal response coordinates make the RCO proxy a sum of per-tensor normalized losses.
            response[i, :, i] = np.sqrt(errors)
            converter.log(f'{t.name} · normalized calibration MSE {errors}')
        if available >= high:
            selected = [min(range(len(bits_options)), key=lambda b: response[i, b, i]) for i in range(len(entries))]
        elif available == low:
            selected = [0] * len(entries)
        else:
            unit = high - low
            norm_costs = (np.asarray(costs, dtype=np.float64) - np.asarray(costs)[:, :1]) / unit
            search_budget = np.clip((available - low) / unit, 1e-6, 1 - 1e-6)
            converter.stage('Поиск RCO')
            raw = dispatch(converter.run, binaries, root, 'rco',
                           np.concatenate((norm_costs.ravel(), response.ravel())), len(entries), 64, len(entries),
                           len(bits_options), 1729, search_budget)
            if abs(float(raw[len(entries)]) - search_budget) > 1e-5:
                raise RuntimeError('RCO could not attain the continuous Vulkan budget.')
            scores = raw[len(entries) + 4:].reshape(-1, len(bits_options))
            feasible = (choices for choices in product(range(len(bits_options)), repeat=len(entries))
                        if sum(row[b] for row, b in zip(costs, choices)) <= available)
            # The final budget test uses exact integer bytes, including at floating-point ties.
            selected = max(feasible, key=lambda choices: sum(row[b] for row, b in zip(scores, choices)))
        if model_loss:
            feasible = (choices for choices in product(range(len(bits_options)), repeat=len(entries))
                        if sum(row[b] for row, b in zip(costs, choices)) <= available)
            alternate = min((c for c in feasible if c != tuple(selected)),
                            key=lambda c: sum(float(response[i, b, i]) ** 2 for i, b in enumerate(c)), default=None)
            candidates = [(selected, variants), (selected, rounded)]
            if alternate is not None:
                candidates.append((alternate, variants))
            return choose_render(converter, binaries, root, base, output, capture, entries, types,
                                 settings, payload, candidates, gguf_module)
        mapping = {t.name: (variants[i][b], types[b]) for i, ((t, _), b) in enumerate(zip(entries, selected))}
        converter.log(f'RCO tensor payload {retained + sum(row[b] for row, b in zip(costs, selected)):,} / {budget:,} bytes')
        return converter.write_mixed_gguf(base, output, mapping)


def choose_render(converter, binaries, root, base, output, recipe, entries, types,
                  settings, payload, candidates, gguf_module):
    from native_capture import pixels, render

    # The reference and candidates must have the same graph execution path.
    # Activation collection can split GPU graph submissions near the captured node.
    converter.stage('Рендер исходной модели')
    expected = pixels(render(converter, settings, payload, root / 'eval-reference'))
    selected = None
    best = float('inf')
    files = {}

    def replace_weights(path, slots, weights):
        with path.open('r+b', buffering=0) as stream:
            for name, (data, _) in weights.items():
                offset, size = slots[name]
                if data.nbytes != size:
                    raise ValueError('Размер тензора не совпадает с местом в рабочем GGUF.')
                stream.seek(offset)
                data.tofile(stream)

    for index, (choices, weights) in enumerate(candidates):
        key = tuple(choices)
        mapping = {t.name: (weights[i][b], types[b]) for i, ((t, _), b) in enumerate(zip(entries, choices))}
        if key in files:
            candidate, slots = files[key]
            replace_weights(candidate, slots, mapping)
        else:
            candidate = root / f'candidate-{index}.gguf'
            converter.write_mixed_gguf(base, candidate, mapping)
            reader = gguf_module('gguf_reader').GGUFReader(candidate, 'r')
            slots = {t.name: (t.data_offset, t.n_bytes) for t in reader.tensors if t.name in mapping}
            del reader
            files[key] = candidate, slots
        config = dict(settings)
        config[recipe.get('base_part', 'sdmodel')] = str(candidate)
        converter.stage(f'Рендер кандидата {index + 1} из {len(candidates)}', index, len(candidates))
        actual = pixels(render(converter, config, payload, root / f'eval-{index}'))
        if actual.shape != expected.shape:
            raise ValueError('Калибровочные рендеры различаются по размеру.')
        loss = 0.
        total = expected.size
        for offset in range(0, total, 4096 * BLOCK):
            a = expected.ravel()[offset:offset + 4096 * BLOCK].astype(np.float32) / 255
            b = actual.ravel()[offset:offset + 4096 * BLOCK].astype(np.float32) / 255
            pad = -len(a) % BLOCK
            paired = np.column_stack((np.pad(a, (0, pad)).reshape(-1, BLOCK),
                                      np.pad(b, (0, pad)).reshape(-1, BLOCK)))
            parts = dispatch(converter.run, binaries, root, 'loss', paired, len(paired), 1, 1, 2, 1729)
            loss += float(parts.sum(dtype=np.float64))
        loss /= total
        converter.log(f'Рендер {index + 1}/{len(candidates)} · MSE {loss:.8g}')
        if loss < best:
            selected, best = (candidate, slots, mapping), loss
    with converter.lock:
        if converter.cancelled:
            raise InterruptedError('Конвертация отменена.')
        winner, slots, weights = selected
        replace_weights(winner, slots, weights)
        winner.rename(output)
    converter.log(f'Готово: {output}\nОшибка модели: {best:.8g}. Размер: {output.stat().st_size:,} байт.')
    return output
