"""Activation-weighted Vulkan calibration with fixed GGUF layouts."""
from contextlib import ExitStack
from pathlib import Path
import tempfile

import numpy as np


LAYOUTS = {'Q4_0': (4, 18), 'Q8_0': (8, 34), 'Q2_K': (2, 84), 'Q3_K': (3, 110),
           'Q4_K': (4, 144), 'Q5_K': (5, 176), 'Q6_K': (6, 210)}
K_SCALES = {'Q2_K': (80, 82), 'Q3_K': (108,), 'Q4_K': (0, 2),
            'Q5_K': (0, 2), 'Q6_K': (208,)}
PREFIXES = ('model.diffusion_model.', 'diffusion_model.')
FLOATS = {'F32', 'F16', 'BF16', 'F8_E4M3', 'F8_E5M2'}


def reference_rows(path, base, gguf_module, stack):
    """Resolve known tensor aliases and return float rows, with explicit scale metadata."""
    if path.suffix.lower() == '.gguf':
        reader = gguf_module('gguf_reader').GGUFReader(path, 'r')
        a, b = base.get_field('general.architecture'), reader.get_field('general.architecture')
        if a is None or b is None or a.contents() != b.contents() or reader.endianess.name != 'LITTLE':
            raise ValueError('Архитектура или порядок байтов исходного GGUF отличается.')
        names = {t.name:t for t in reader.tensors}
        decode = gguf_module('quants').dequantize
        def gguf_rows(t, part):
            ref = names.get(t.name)
            if ref is None or tuple(ref.shape) != tuple(t.shape):
                raise ValueError(f'Исходный GGUF не содержит совпадающий тензор {t.name}.')
            return decode(ref.data[part], ref.tensor_type).astype(np.float32, copy=False)
        return gguf_rows
    if path.suffix.lower() not in ('.safetensors', '.sft'):
        raise ValueError('Используйте GGUF или Safetensors в качестве исходника калибровки.')
    from moss_convert import SafeFile, decode_rows
    ref = SafeFile(path)
    stack.callback(ref.data._mmap.close)
    def safe_rows(t, part):
        short=t.name
        for prefix in PREFIXES:
            short=short.removeprefix(prefix)
        aliases=[t.name, short, *(p+short for p in PREFIXES)]
        if t.name.startswith('model.'):
            aliases.append('model.language_model.'+t.name.removeprefix('model.'))
        key=next((name for name in aliases if name in ref.header), None)
        if key is None:
            raise ValueError(f'В исходных весах не найден {t.name}.')
        info=ref.header[key]
        layer=key.removesuffix('.weight')
        scheme=ref.formats.get(layer,{})
        form=scheme.get('format','')
        packed=form=='asym_w4a8_int8'
        shape=info['shape']
        wanted=list(reversed(t.shape.tolist()))
        if len(shape)!=2 or wanted!=[shape[0], shape[1]*(2 if packed else 1)]:
            raise ValueError(f'Размеры исходного и целевого тензоров не совпадают у {t.name}.')
        if form not in ('','asym_w4a8_int8','int8_tensorwise','float8_e4m3fn','float8_e5m2'):
            raise ValueError(f'Для {key} нужен декодер схемы {form}.')
        if packed:
            if info['dtype'] not in ('I8','U8') or not all(layer+s in ref.header for s in ('.weight_codebook','.weight_s_rel','.weight_s_channel')):
                raise ValueError(f'У {key} нет полного набора метаданных INT4 W4A8.')
        elif info['dtype']=='I8':
            if layer+'.weight_scale' not in ref.header:
                raise ValueError(f'Для {key} нет масштаба INT8.')
        elif info['dtype'] not in FLOATS:
            raise ValueError(f'Неизвестный формат исходника {info["dtype"]} у {key}.')
        if any(layer+s in ref.header for s in ('.weight_scale_inv','.scale_weight')):
            raise ValueError(f'У {key} нужен поддерживаемый формат масштаба weight_scale.')
        return decode_rows(ref,key,part,scheme)
    return safe_rows


def solve(converter, root, helper, weights, importance, old, bits, steps):
    """Vulkan blockwise weighted scale/code search with host packing of FP16 scales."""
    blocks=len(old)
    raw=old.reshape(blocks,-1)
    scales=raw[:,:2].copy().view('<f2').ravel()
    data=root/'input.bin'
    result=root/'output.bin'
    with data.open('wb') as f:
        weights.astype('<f4',copy=False).tofile(f)
        importance.astype('<f4',copy=False).tofile(f)
        scales.astype('<f4').tofile(f)
    converter.run(helper,'actcal',data,result,helper.with_name('actcal.spv'),
                  blocks,steps,len(importance)//32,bits,0,0,1)
    words=5 if bits==4 else 9
    encoded=np.fromfile(result,dtype='<u4')
    if encoded.size!=blocks*words:
        raise RuntimeError('Неполный результат Vulkan-калибровки.')
    encoded=encoded.reshape(blocks,words)
    scale=encoded[:,0].copy().view('<f4').astype('<f2')
    if not np.isfinite(scale).all():
        raise RuntimeError('Неконечный масштаб Vulkan-калибровки.')
    return np.concatenate((scale.reshape(-1,1).view(np.uint8),encoded[:,1:].copy().view(np.uint8)),axis=1)


def solve_k(converter, root, helper, weights, importance, old, kind, decode):
    """Fit the FP16 superblock scales to the existing K-quants and subscale codes."""
    offsets = K_SCALES[kind.name]
    basis = old.copy()
    data, result = root/'input.bin', root/'output.bin'
    with data.open('wb') as f:
        weights.astype('<f4',copy=False).tofile(f)
        for unit in offsets:
            for offset in offsets:
                basis[:,offset:offset+2] = (0,60 if offset==unit else 0)  # FP16 1.0 or 0.0
            decode(basis,kind).astype('<f4',copy=False).tofile(f)
        importance.astype('<f4',copy=False).tofile(f)
    converter.run(helper,'actcal-k',data,result,helper.with_name('actcal_k.spv'),
                  len(old),1,len(importance)//256,len(offsets),0,0,1)
    scale=np.fromfile(result,dtype='<f4')
    if scale.size!=len(old)*len(offsets) or not np.isfinite(scale).all():
        raise RuntimeError('Некорректные масштабы Vulkan для K-кванта.')
    packed=old.copy()
    encoded=np.clip(scale.reshape(-1,len(offsets)),-65504,65504).astype('<f2')
    for i,offset in enumerate(offsets):
        packed[:,offset:offset+2] = encoded[:,i].copy().view(np.uint8).reshape(-1,2)
    return packed


def optimize_full(converter, manifest, base, reference, target, output, gguf_module, *, capture, detect,
                  evidence=None, binaries=None):
    if base.endianess.name!='LITTLE':
        raise ValueError('Требуется little-endian GGUF.')
    steps=target.get('steps',3)
    if type(steps) is not int or not 1<=steps<=8:
        raise ValueError('Калибровка принимает от одного до восьми шагов.')
    specs=target.get('tensors')
    all_weights=target.get('all') is True
    if not all_weights and (not isinstance(specs,dict) or not specs):
        raise ValueError('Нужен список тензоров/активаций или all: true для автозаписи.')
    weights={t.name:t for t in base.tensors}
    if specs and set(specs)-weights.keys():
        raise ValueError('В манифесте есть неизвестные тензоры.')
    entries=[t for t in base.tensors if (all_weights or t.name in specs)
             and t.tensor_type.name in LAYOUTS and len(t.shape)==2
             and int(t.shape[0]) % (256 if t.tensor_type.name in K_SCALES else 32) == 0]
    if not entries or not all_weights and len(entries)!=len(specs):
        raise ValueError('Нужны матрицы Q4_0/Q8_0 или Q2_K–Q6_K с подходящей шириной блока.')
    helper=(binaries or Path(__file__).resolve().parent/'vulkan')/'vulkan-gsq.exe'
    decode=gguf_module('quants').dequantize
    with tempfile.TemporaryDirectory(prefix='.gguf-full-cal-',dir=output.parent) as temp, ExitStack() as maps:
        root=Path(temp)
        ref=reference_rows(reference,base,gguf_module,maps)
        found={}
        automatic=[t.name for t in entries if all_weights or specs[t.name] is None]
        if automatic:
            from native_capture import collect
            directory=root/'activation'
            directory.mkdir()
            found,_,_=collect(converter,manifest,Path(base.data.filename),capture,automatic,
                               directory,detect,require_all=False)
        replacements={}
        measured={}
        for index,t in enumerate(entries):
            path=found.get(t.name)
            if path is None:
                if all_weights or specs[t.name] is None:
                    continue
                relative=specs[t.name]
                if not isinstance(relative,str) or Path(relative).suffix!='.npy':
                    raise ValueError(f'{t.name}: нужны активации .npy.')
                path=(manifest.parent/relative).resolve(strict=True)
            x=np.load(path,allow_pickle=False)
            width,rows=map(int,t.shape)
            if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[1]!=width or not 8<=len(x)<=32 or not np.isfinite(x).all():
                raise ValueError(f'{t.name}: ожидаются 8–32 строки активаций шириной {width}.')
            x=x.astype(np.float32)
            count=len(x)*3//4
            importance=(x[:count]**2).mean(axis=0)
            if float(importance.max())==0:
                continue
            importance=np.maximum(importance/importance.max(),1e-6)
            holdout=np.ascontiguousarray(x[count:].T)
            candidate=np.memmap(root/f'{index}.bin',mode='w+',dtype='uint8',shape=t.data.shape)
            maps.callback(candidate._mmap.close)
            bits,packed_bytes=LAYOUTS[t.tensor_type.name]
            chunk=max(1,8*1024*1024//width)
            before=after=0.0
            converter.stage(f'Калибровка {index+1}/{len(entries)} · {t.tensor_type.name}',0,rows)
            for start in range(0,rows,chunk):
                part=slice(start,min(rows,start+chunk))
                reference_values=np.ascontiguousarray(ref(t,part),dtype=np.float32)
                n=len(reference_values)
                old=t.data[part].reshape(-1,packed_bytes)
                if t.tensor_type.name in K_SCALES:
                    packed=solve_k(converter,root,helper,reference_values,importance,old,t.tensor_type,decode)
                else:
                    packed=solve(converter,root,helper,reference_values,importance,old,bits,steps)
                expected=reference_values @ holdout
                old_loss=((decode(old,t.tensor_type).reshape(n,width) @ holdout-expected)**2).sum(axis=1)
                new_loss=((decode(packed,t.tensor_type).reshape(n,width) @ holdout-expected)**2).sum(axis=1)
                choose=(new_loss < old_loss)
                old_rows=old.reshape(n,-1)
                new_rows=packed.reshape(n,-1)
                candidate[part]=np.where(choose[:,None],new_rows,old_rows)
                before+=float(old_loss.sum(dtype=np.float64))
                after+=float(np.minimum(old_loss,new_loss).sum(dtype=np.float64))
                converter.stage(f'Калибровка {index+1}/{len(entries)} · {t.tensor_type.name}',part.stop,rows)
            measured[t.name]=dict(before=before,after=after,changed=after<before,bytes=t.n_bytes)
            if after<before:
                replacements[t.name]=candidate,t.tensor_type
            converter.log(f'{t.name} · {t.tensor_type.name} · {before:.6g} → {after:.6g}')
        if not measured:
            raise ValueError('В калибровочном запросе не было пригодных активаций для выбранных матриц.')
        summary=dict(eligible=len(entries),observed=len(measured),changed=len(replacements),
                     unobserved=[t.name for t in entries if t.name not in measured],tensors=measured)
        if evidence is not None:
            evidence.update(summary)
        converter.log(f'Матриц обработано {len(measured)}/{len(entries)}, улучшилось {len(replacements)}.')
        return converter.write_mixed_gguf(base,output,replacements,preserve_file_type=True)
