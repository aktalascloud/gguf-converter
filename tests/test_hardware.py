"""Check memory recommendations without assuming a specific GPU."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hardware import GIB, Hardware, advice, safetensor_sizes

card=Hardware(('AVX','AVX2'),'Example',8*GIB,7*GIB,True,True)
sizes={'Q2':2*GIB,'Q4':4*GIB,'Q6':5*GIB,'Q8':7*GIB}
assert card.cpu_target=='avx2' and card.gpu_target=='polaris'
assert advice(sizes,card)==('Q6',None)
assert advice(sizes,card,llm=True)==(None,None)
tier,context=advice(sizes,card,llm=True,geometry=(128*1024,8192))
assert tier=='Q6' and 2048 <= context <= 8192 and context%256==0
assert advice(sizes,Hardware(vram_bytes=8*GIB,budget_bytes=0))==(None,None)
assert safetensor_sizes({'a.weight':{'shape':[16,256],'dtype':'F16','data_offsets':[0,8192]}},
                       {'Q6':(256,210)})=={'Q6':3360}
assert safetensor_sizes({'a.weight':{'shape':[16,256],'dtype':'I8','data_offsets':[0,4096]}},
                       {'Q6':(256,210)})=={}
