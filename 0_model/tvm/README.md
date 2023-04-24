### Requirements

* tvm v0.10.0

### How to run TVM

- Run `bash run.sh`. For V100, modify the `sm_80` in the script to `sm_70`.
- Longformer: since G2BMM and GBMM are not native supported by TVM. We split the model into three parts, i.e., G2BMM + GBMM, the head, and the tail of model. Run `./longformer/run.sh` to get the time of head and tail. The time of GBMM + G2BMM is evluated in A.5.2. In our evaluation, the execution time of these two operators are shown in the following table. 

| G2BMM+GBMM (ms) | A100    | V100     |
|-----------------|---------|----------|
| Batch size 1    | 11.5681 |  34.3496 |
| Batch size 16   | 182.451 | 542.7025 |
