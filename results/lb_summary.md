# MABe leaderboard simulation — summary

Protocol: multi-lab corpus (9 labs, the 7-keypoint CalMS21 schema), split by
video and stratified by lab, **official** metric (F1 per action -> mean per
lab -> mean across labs). Every block is run over both **raw** tracking and
tracking **corrected with MouseSkeleton**.

## Models (official metric)

```
condition    corr     raw  delta_corr_raw
model                                    
bilstm     0.1877  0.2273         -0.0396
cnn_lstm   0.1885  0.2215         -0.0330
gru        0.1880  0.2104         -0.0224
histgb     0.1822  0.2241         -0.0419
rf         0.0576  0.0842         -0.0266
tcn        0.2212  0.2455         -0.0243
```

Best: **raw / tcn** = 0.2455

## Public (32%) vs private (68%)

```
condition    model  score_full  score_public_32  score_private_68  boot_ci_lo  boot_ci_hi  boot_range
      raw      tcn      0.2455           0.2744            0.2438      0.1670      0.3614      0.1944
      raw   bilstm      0.2273           0.2637            0.2544      0.1619      0.3510      0.1891
      raw   histgb      0.2241           0.2900            0.2351      0.1528      0.3493      0.1965
      raw cnn_lstm      0.2215           0.2554            0.2547      0.1586      0.3432      0.1846
     corr      tcn      0.2212           0.2815            0.2263      0.1448      0.3409      0.1961
      raw      gru      0.2104           0.2487            0.2236      0.1430      0.3362      0.1932
     corr cnn_lstm      0.1885           0.2267            0.2010      0.1195      0.3178      0.1983
     corr      gru      0.1880           0.2224            0.2046      0.1251      0.3174      0.1923
     corr   bilstm      0.1877           0.2443            0.1919      0.1166      0.3197      0.2031
     corr   histgb      0.1822           0.2086            0.1937      0.1173      0.3066      0.1893
      raw       rf      0.0842           0.1033            0.0887      0.0319      0.1745      0.1425
     corr       rf      0.0576           0.1020            0.0658      0.0214      0.1534      0.1320
```

- |public - private|: mean 0.0281, max 0.0552
- Mean width of the public 95% CI: 0.1843
- Spread between models: {'corr': 0.1636, 'raw': 0.1613}

> When the CI width exceeds the spread between models, the public
> leaderboard ordering does not distinguish architectures.

## O1 - Geometry

Violations: raw **0.1747** -> corr **0.1209** (-5.38 p.p., 27 videos)

## O3 - Active learning (AULC)

```
condition         strategy  aulc_mean  aulc_std  delta_vs_random  wilcoxon_p
     corr            badge     0.0426    0.0034          -0.0035      0.4375
     corr          coreset     0.0441    0.0071          -0.0020      0.4375
     corr          entropy     0.0400    0.0071          -0.0061      0.1250
     corr least_confidence     0.0386    0.0091          -0.0075      0.0625
     corr           margin     0.0465    0.0027           0.0004      1.0000
     corr           random     0.0461    0.0074           0.0000         NaN
      raw            badge     0.0504    0.0074           0.0000      1.0000
      raw          coreset     0.0505    0.0045           0.0002      1.0000
      raw          entropy     0.0439    0.0059          -0.0064      0.0625
      raw least_confidence     0.0418    0.0091          -0.0086      0.1875
      raw           margin     0.0462    0.0062          -0.0041      0.3125
      raw           random     0.0503    0.0029           0.0000         NaN
```

Strategies significantly beating random: **none**

## O4 - Cross-lab (mean off the diagonal)

```
condition               corr     raw
train_lab                           
CRIM13                0.0390  0.0426
CalMS21_supplemental  0.0518  0.0345
CalMS21_task1         0.0579  0.0835
CalMS21_task2         0.0773  0.0757
CautiousGiraffe       0.0000  0.0000
ElegantMink           0.0000  0.0046
InvincibleJellyfish   0.0733  0.0607
JovialSwallow         0.0015  0.0025
TranquilPanther       0.0000  0.0000
```

Mean action overlap between different labs: 2.25

## Computational cost

```
         videos  frames   total_s  ms_per_frame
variant                                        
corr         89  768714  20776.54         27.05
raw          90  768714    111.55          0.73
```

Training plus inference per model (s):

```
condition    corr     raw
model                    
bilstm     2025.6  2004.0
cnn_lstm   3919.6  3912.2
gru         315.8   314.4
histgb       51.9    54.0
rf           17.2    20.1
tcn        2040.9  2034.6
```
