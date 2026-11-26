| kernel family | before us/step | calls | after us/step | calls |
|---|---:|---:|---:|---:|
| Q8 projections / wo / head GEMV | 4572 | 289 | 3308 | 97 |
| NVFP4 expert GEMV | 2679 | 88 | 2831 | 88 |
| Q8_0 expert GEMV | 877 | 8 | 555 | 8 |
| attention (+KV write, clip, conv) | 1012 | 256 | 1029 | 256 |
| block mixes | 919 | 288 | 600 | 192 |
| loop control | 997 | 288 | 732 | 144 |
| router (GEMV, top-8, grouping) | 555 | 144 | 388 | 48 |
| tiny BF16 GEMVs (sinks, conv proj) | 240 | 64 | 0 | 0 |
| other small (norms, copies, conv, situ, moe sum, argmax) | 718 | 309 | 635 | 261 |
| **total** | **12569** | **1734** | **10078** | **1094** |
