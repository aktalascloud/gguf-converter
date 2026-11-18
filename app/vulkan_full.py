"""Whole-checkpoint GSQ and byte-constrained RCO on Vulkan."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from activation_calibration import reference_rows
from vulkan_workflow import BLOCK, BLOCK_BYTES, dispatch, pack_k, rtn


VARIANTS = {
    'Q4_0': ('Q3_K', 'Q4_0', 'Q6_K'),
    'Q4_1': ('Q4_0', 'Q4_1', 'Q6_K'),
    'Q4_K': ('Q3_K', 'Q4_K', 'Q6_K'),
    'Q6_K': ('Q4_K', 'Q6_K', 'Q8_0'),
    'Q8_0': ('Q4_0', 'Q6_K', 'Q8_0'),
}
NARROW = ('Q4_0', 'Q4_1', 'Q8_0')


def row_bytes(kind, width):
    size = {'Q4_0': 144, 'Q4_1': 160}.get(kind.name, BLOCK_BYTES[int(kind.name[1])])
    return width * size // BLOCK


def pack(codes, scales, kind, center):
    if kind.name != 'Q4_1':
        return pack_k(codes, scales, int(kind.name[1]), legacy_q4=kind.name == 'Q4_0')
    scale = scales.astype('<f2')
    minimum = (center - 8 * scale.astype(np.float32)).astype('<f2')
    if not np.isfinite(scale).all() or not np.isfinite(minimum).all():
        raise ValueError('Q4_1 scale or offset is outside the FP16 range.')
    pairs = (codes + 8).astype(np.uint8).reshape(-1, 8, 2, 16)
    data = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    return np.concatenate((scale.view(np.uint8).reshape(-1, 8, 2),
                           minimum.view(np.uint8).reshape(-1, 8, 2), data), axis=2).reshape(-1, 160)


def fit_rows(converter, binaries, work, ref, stock, kind, train, valid, decode, steps, seed):
    rows, width = ref.shape
    original_width = width
    pad = (-width) % BLOCK
    if pad:
        ref, train, valid = (np.pad(x, ((0, 0), (0, pad))) for x in (ref, train, valid))
        if stock is not None:
            stock = np.pad(stock, ((0, 0), (0, row_bytes(kind, pad))))
        width += pad
    ncols = width // BLOCK
    weights = ref.reshape(-1, BLOCK)
    center = None
    if kind.name == 'Q4_1':
        values = weights.reshape(-1, 8, 32)
        center = (values.max(axis=2) + values.min(axis=2)) * 0.5
        weights = weights - np.repeat(center, 32, axis=1)
    activation = train.reshape(len(train), ncols, BLOCK).transpose(1, 0, 2)
    payload = np.concatenate((weights.ravel(), activation.ravel()))
    bits, n = int(kind.name[1]), len(weights)
    raw = dispatch(converter.run, binaries, work, 'gsq_shared', payload, n, steps, len(train), bits, seed,
                   budget=ncols)
    learnt = pack(raw[:n*BLOCK].reshape(n, BLOCK), raw[n*258:].reshape(n, 8), kind, center)
    rounded = pack(*rtn(weights, bits), kind, center) if stock is None else stock.reshape(n, -1)
    old = decode(rounded, kind).reshape(rows, ncols, BLOCK)
    new = decode(learnt, kind).reshape(rows, ncols, BLOCK)
    expected = ref.reshape(rows, ncols, BLOCK)
    probes = activation.transpose(0, 2, 1)
    old_error = ((old - expected)[..., None, :] @ probes).reshape(rows, ncols, -1)
    new_error = ((new - expected)[..., None, :] @ probes).reshape(rows, ncols, -1)
    improved = np.sum(new_error**2, axis=-1) < np.sum(old_error**2, axis=-1)
    selected = np.where(improved.reshape(n, 1), learnt, rounded).reshape(rows, -1)
    output = np.where(improved[..., None], new, old).reshape(rows, width)
    error = ((output - ref) @ valid.T)**2
    return selected[:, :row_bytes(kind, original_width)], float(error.sum(dtype=np.float64))


def choose(converter, binaries, root, costs, errors, budget):
    """Project continuous RCO scores to a feasible discrete byte assignment."""
    group = np.arange(len(costs))
    minimum, maximum = int(costs[:, 0].sum()), int(costs[:, -1].sum())
    if budget < minimum:
        raise ValueError(f'Tensor budget requires at least {minimum:,} eligible bytes.')
    if budget == minimum:
        return np.zeros(len(group), dtype=int), None
    if budget >= maximum:
        return np.argmin(errors, axis=1), None
    relative_cost = (costs - costs[:, :1]) / (maximum - minimum)
    target = (budget - minimum) / (maximum - minimum)
    losses = errors / max(float(errors.max()), 1e-20)
    raw = dispatch(converter.run, binaries, root, 'rco_full',
                   np.concatenate((relative_cost.ravel(), losses.ravel())), len(group), 16, 1, 3, 1729, target)
    residual = float(raw[len(group)+3])
    if residual > 1e-5 or abs(float(raw[len(group)])-target) > 1e-5:
        raise RuntimeError('Vulkan RCO did not retract to the requested budget.')
    scores = raw[len(group)+4:].reshape(-1, 3).astype(np.float64)

    def allocation(penalty):
        return np.argmax(scores - penalty * relative_cost, axis=1)

    selected = allocation(0)
    if int(costs[group, selected].sum()) > budget:
        lo, hi = 0., 1.
        while int(costs[group, allocation(hi)].sum()) > budget:
            hi *= 2
        for _ in range(48):
            mid = (lo + hi) * .5
            if int(costs[group, allocation(mid)].sum()) > budget:
                lo = mid
            else:
                hi = mid
        selected = allocation(hi)

    # Keep the actual serialized payload under the exact integer budget.
    left = budget - int(costs[group, selected].sum())
    improvements = []
    for g, current in enumerate(selected):
        for variant in range(3):
            delta = int(costs[g, variant] - costs[g, current])
            gain = errors[g, current] - errors[g, variant]
            if gain > 0 and delta >= 0:
                improvements.append((-gain / max(delta, 1), g, current, variant, delta))
    for _, g, current, variant, delta in sorted(improvements):
        if current == selected[g] and delta <= left:
            selected[g] = variant
            left -= delta
    assert int(costs[group, selected].sum()) <= budget
    return selected, residual


def optimize(converter, manifest, base, reference, target, output, gguf_module, *, binaries=None, evidence=None):
    specs = target.get('tensors')
    budget, steps = target.get('max_tensor_bytes'), target.get('steps', 2)
    if base.endianess.name != 'LITTLE' or not isinstance(specs, dict) or not 1 <= len(specs) <= 512:
        raise ValueError('Full Vulkan GSQ/RCO needs a little-endian GGUF and 1–512 recorded activations.')
    if type(budget) is not int or budget <= 0 or type(steps) is not int or not 1 <= steps <= 16:
        raise ValueError('Set an integer tensor byte budget and 1–16 full GSQ steps.')
    all_weights = {t.name: t for t in base.tensors}
    quants = gguf_module('quants')
    qtypes = gguf_module('constants').GGMLQuantizationType
    binaries = binaries or Path(__file__).resolve().parent / 'vulkan'
    costs, entries = [], []
    for name, relative in specs.items():
        t = all_weights.get(name)
        if t is None or len(t.shape) != 2 or not 32 <= int(t.shape[0]) <= BLOCK * 4096 or int(t.shape[0]) % 32:
            raise ValueError(f'{name}: expected a two-dimensional weight with a width divisible by 32.')
        if t.tensor_type.name not in VARIANTS or not isinstance(relative, str) or Path(relative).suffix.lower() != '.npy':
            raise ValueError(f'{name}: expected a supported layout and recorded .npy activations.')
        path = (manifest.parent / relative).resolve(strict=True)
        x = np.load(path, allow_pickle=False)
        if not isinstance(x, np.ndarray):
            x.close()
            raise ValueError(f'{name}: expected a NumPy array.')
        x = x.astype(np.float32, copy=False)
        if x.ndim != 2 or x.shape[1] != t.shape[0] or not 8 <= len(x) <= 32 or not np.isfinite(x).all():
            raise ValueError(f'{name}: expected 8–32 finite activation rows matching the tensor width.')
        width = int(t.shape[0])
        if width % BLOCK and t.tensor_type.name not in NARROW:
            raise ValueError(f'{name}: the chosen K-quant layout needs width divisible by 256.')
        kinds = tuple(qtypes[k] for k in (NARROW if width % BLOCK else VARIANTS[t.tensor_type.name]))
        costs.append([int(t.shape[1]) * row_bytes(k, width) for k in kinds])
        entries.append((t, kinds, x[:min(8, len(x)*3//4)], x[len(x)*3//4:]))
    retained = sum(t.n_bytes for t in base.tensors if t.name not in specs)
    costs = np.asarray(costs, dtype=np.int64)
    if budget - retained < int(costs[:, 0].sum()):
        raise ValueError('The requested budget cannot fit the least costly layouts.')

    with TemporaryDirectory(prefix='.gguf-full-gsq-', dir=output.parent) as directory, ExitStack() as maps:
        root = Path(directory)
        read = reference_rows(reference, base, gguf_module, maps)
        errors = np.zeros(costs.shape, dtype=np.float64)
        for i, (t, kinds, train, valid) in enumerate(entries):
            rows = int(t.shape[1])
            indices = np.unique(np.linspace(0, rows-1, min(32, rows), dtype=int))
            ref = np.concatenate([read(t, slice(j, j+1)) for j in indices])
            energy = max(float(((ref @ valid.T)**2).sum(dtype=np.float64)), 1e-30)
            old = t.data[indices]
            for k, kind in enumerate(kinds):
                stock = old if kind == t.tensor_type else None
                _, error = fit_rows(converter, binaries, root, ref, stock, kind, train, valid, quants.dequantize,
                                    steps, 1729+i*3+k)
                errors[i, k] = error / energy
            converter.stage('GSQ/RCO · оценка типов', i+1, len(entries))
        selected, residual = choose(converter, binaries, root, costs, errors, budget - retained)
        replacements, fit = {}, {}
        for i, ((t, kinds, train, valid), choice) in enumerate(zip(entries, selected)):
            kind = kinds[choice]
            width, rows = map(int, t.shape)
            batch = max(1, min(16384, 32768//steps) // ((width+BLOCK-1)//BLOCK))
            out = np.memmap(root/f'weight-{i}.bin', dtype=np.uint8, mode='w+',
                            shape=(rows, row_bytes(kind, width)))
            maps.callback(out._mmap.close)
            before = after = 0.
            for start in range(0, rows, batch):
                part = slice(start, min(rows, start+batch))
                ref, old = read(t, part), t.data[part]
                stock = old if kind == t.tensor_type else None
                out[part], error = fit_rows(converter, binaries, root, ref, stock, kind, train, valid,
                                            quants.dequantize, steps, (3451+i*65537+start) % (1<<32))
                before += float((((quants.dequantize(old, t.tensor_type)-ref) @ valid.T)**2).sum(dtype=np.float64))
                after += error
            keep = kind != t.tensor_type or after < before
            if keep:
                replacements[t.name] = out, kind
            fit[t.name] = {'stock': t.tensor_type.name, 'selected': kind.name, 'rows': rows,
                           'before': before, 'after': after if keep else before}
            converter.log(f'GSQ {i+1}/{len(entries)} · {t.name} · {kind.name} · {before:.6g} → {fit[t.name]["after"]:.6g}')
            converter.stage('GSQ · все строки', i+1, len(entries))
        payload = retained + int(costs[np.arange(len(entries)), selected].sum())
        if evidence is not None:
            evidence.update(payload=payload, budget=budget, residual=residual, tensors=fit,
                            layouts=dict(Counter(row['selected'] for row in fit.values())))
        converter.log(f'RCO GGUF payload: {payload:,} / {budget:,} bytes.')
        same_layout = all(kind == all_weights[name].tensor_type for name, (_, kind) in replacements.items())
        return converter.write_mixed_gguf(base, output, replacements, preserve_file_type=same_layout)
