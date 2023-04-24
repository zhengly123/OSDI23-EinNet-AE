### how to run Nimble

File `OSDI23-EinNet-AE/0_model/nimble/experiment/evaluation.sh` runs all the experiments for nimble;

Nimble does not support `longformer`

```
# make sure environment var NNET_HOME is set to /path/to/OSDI23-EinNet-AE
# current directory is OSDI23-EinNet-AE/0_model/nimble
source ./env_nimble.sh
bash experiment/evaluation.sh
```