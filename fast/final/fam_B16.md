| kernel family | before us/step | calls | after us/step | calls |
|---|---:|---:|---:|---:|
| Q8 projections / wo / head GEMV | 12731 | 289 | 6867 | 97 |
| NVFP4 expert GEMV | 22406 | 88 | 23164 | 88 |
| Q8_0 expert GEMV | 5711 | 8 | 3727 | 8 |
| attention (+KV write, clip, conv) | 3007 | 256 | 2998 | 256 |
| block mixes | 1085 | 288 | 819 | 192 |
| loop control | 2942 | 288 | 2532 | 144 |
| router (GEMV, top-8, grouping) | 1147 | 144 | 1232 | 144 |
| tiny BF16 GEMVs (sinks, conv proj) | 871 | 64 | 0 | 0 |
| other small (norms, copies, conv, situ, moe sum, argmax) | 1883 | 309 | 1857 | 261 |
| **total** | **51784** | **1734** | **43197** | **1190** |
