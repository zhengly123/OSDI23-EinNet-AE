`evaluate_max_depth.py` produces the results in Figure 15 on the A100 GPU.

This script can be executed by `python3 ./evaluate_max_depth.py`

An expected result should be like

```
Figure 16
=== Model InfoGAN.bs1
InfoGAN.bs1 Depth = 1: 1.3334470000000003 ms
InfoGAN.bs1 Depth = 2: 1.3334470000000003 ms
InfoGAN.bs1 Depth = 3: 0.1114935930053711 ms
InfoGAN.bs1 Depth = 4: 0.0907476697265625 ms
InfoGAN.bs1 Depth = 5: 0.0907476697265625 ms
InfoGAN.bs1 Depth = 6: 0.0907476697265625 ms
=== Model longformer.bs1
longformer.bs1 Depth = 1: 26.658973366666665 ms
longformer.bs1 Depth = 2: 26.658973366666665 ms
longformer.bs1 Depth = 3: 26.658973366666665 ms
longformer.bs1 Depth = 4: 11.243110000000001 ms
longformer.bs1 Depth = 5: 11.243110000000001 ms
longformer.bs1 Depth = 6: 11.243110000000001 ms
```
