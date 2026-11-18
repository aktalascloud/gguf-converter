"""Fixed-scale greedy code refinement with 32-column Hessian blocks on Vulkan.

The update and K code layouts follow huawei-csl/MACaQ's uniform_refine.py
and gguf_codecs.py at 3810efca0b9e819c8eb51addd93c3c87225b7476 (Apache-2.0).
"""
import numpy as np


BITS={'Q4_0':4,'Q4_1':4,'Q8_0':8,'Q4_K':4,'Q5_K':5,'Q6_K':6}


def f16(raw):
    return raw.copy().view('<f2').astype(np.float32)


def unpack(old, layout):
    if layout in ('Q4_0','Q4_1','Q8_0'):
        raw=old[:,4 if layout=='Q4_1' else 2:]
        codes=raw^128 if layout=='Q8_0' else np.concatenate((raw&15,raw>>4),axis=1)
        return codes,np.repeat(f16(old[:,:2]),32,axis=1)
    if layout=='Q6_K':
        low=old[:,:128].reshape(-1,2,64)
        high=old[:,128:192].reshape(-1,2,32)
        lo=np.stack((low&15,low>>4),axis=2).reshape(-1,2,128)
        hi=np.stack([(high>>j)&3 for j in (0,2,4,6)],axis=2).reshape(-1,2,128)
        scales=f16(old[:,208:210])*old[:,192:208].view(np.int8)
        return (lo|(hi<<4)).reshape(-1,256),np.repeat(scales,16,axis=1)
    low=old[:,16 if layout=='Q4_K' else 48:].reshape(-1,4,32)
    codes=np.stack((low&15,low>>4),axis=2).reshape(-1,8,32)
    if layout=='Q5_K':
        high=(old[:,None,16:48]>>np.arange(8,dtype=np.uint8)[None,:,None])&1
        codes|=high<<4
    small, high=old[:,4:8],old[:,12:16]
    scales=np.concatenate((small&63,(high&15)|((small>>2)&48)),axis=1)
    return codes.reshape(-1,256),np.repeat(f16(old[:,:2])*scales,32,axis=1)


def pack(old, codes, layout):
    result=old.copy()
    if layout=='Q8_0':
        result[:,2:]=codes^128
    elif layout in ('Q4_0','Q4_1'):
        result[:,4 if layout=='Q4_1' else 2:]=codes[:,:16]|(codes[:,16:]<<4)
    elif layout=='Q6_K':
        low=(codes&15).reshape(-1,2,2,64)
        hi=(codes>>4).reshape(-1,2,4,32)
        result[:,:128]=(low[:,:,0]|(low[:,:,1]<<4)).reshape(-1,128)
        result[:,128:192]=sum(hi[:,:,j]<<(2*j) for j in range(4)).reshape(-1,64)
    else:
        low=(codes&15).reshape(-1,4,2,32)
        result[:,16 if layout=='Q4_K' else 48:]=(low[:,:,0]|(low[:,:,1]<<4)).reshape(-1,128)
        if layout=='Q5_K':
            hi=(codes>>4).reshape(-1,8,32)
            result[:,16:48]=sum(hi[:,j]<<j for j in range(8))
    return result


def hessian(activations):
    # ponytail: a full-width Hessian exceeds the working set on older Vulkan cards.
    blocks=activations.reshape(len(activations),-1,32).transpose(1,0,2)
    return (blocks.transpose(0,2,1)@blocks)/len(activations)


def refine(converter,root,helper,weights,old,kind,decode,curvature,steps):
    codes, scales=unpack(old,kind.name)
    residual=decode(old,kind).reshape(weights.shape)-weights
    if not np.any(residual):
        return old
    path,result=root/'input.bin',root/'output.bin'
    with path.open('wb') as f:
        for values in (residual,scales,codes,curvature):
            values.astype('<f4',copy=False).tofile(f)
    converter.run(helper,'macaq-flip',path,result,helper.with_name('macaq_flip.spv'),
                  codes.size//32,steps,len(curvature),BITS[kind.name],0,0,1)
    encoded=np.fromfile(result,dtype=np.uint8)
    if encoded.size!=codes.size or (BITS[kind.name]<8 and np.any(encoded >= 1<<BITS[kind.name])):
        raise RuntimeError('Vulkan вернул повреждённые коды MACaQ-R.')
    return pack(old,encoded.reshape(codes.shape),kind.name)
