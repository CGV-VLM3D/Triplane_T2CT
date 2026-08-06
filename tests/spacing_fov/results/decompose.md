# E5 — 2x2 decomposition (n=100, same seed per case)

metric                               A                     B                     C                     D
FID_2p5D_Avg                     7.347                 1.636                 1.896                10.008
FID_2p5D_XY                      3.734                 1.742                 1.962                 4.546
FID_2p5D_YZ                     10.052                 1.224                 1.336                14.270
FID_2p5D_XZ                      8.256                 1.941                 2.388                11.208
CLIPScore                       56.556                51.881                62.717                43.253
CLIPScore_I2I                   42.870                40.468                44.088                32.802
FVD_CTCLIP                       0.294                 0.284                 0.291                 0.502

## Effect sizes on FID_2p5D_Avg
declaration only (A→B, gen@1.0 relabelled to 0.8): -5.711
declaration only (D→C, gen@0.8 relabelled to 0.8): -8.112
conditioning only (A→D, same 1.0 label):           +2.661
conditioning only (B→C, same 0.8 label):           +0.260
total observed  (A→C):                             -5.452
