Final return = mean of the last 3 evaluations (1M env steps), mean [95% stratified bootstrap CI] over seeds.

| Method | cartpole-swingup | quadruped-walk | walker-walk | walker-run | Aggregate |
|---|---|---|---|---|---|
| KalMamba | 530 [523, 537] (n=4) | 830 [670, 919] (n=4) | 898 [890, 909] (n=4) | 432 [389, 456] (n=4) | 673 [633, 699] |
| RSSM+SAC | 498 [484, 510] (n=4) | 775 [484, 928] (n=4) | 956 [952, 959] (n=4) | 500 [431, 537] (n=4) | 682 [608, 729] |
| VRKN+SAC | 691 [661, 724] (n=4) | 930 [916, 944] (n=4) | 881 [720, 963] (n=4) | 560 [550, 566] (n=4) | 765 [724, 792] |
| SAC | 243 [164, 295] (n=4) | 70 [48, 91] (n=4) | 511 [499, 522] (n=4) | 194 [176, 209] (n=4) | 254 [235, 269] |
