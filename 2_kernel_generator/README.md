This folder prodcues the data in Figure 15 on the A100 GPU.

`bash ./run.sh` runs both Ansor and AutoTVM. Since, tuning kenerls are time-consuming and occupies computation resources, we provide tuning records in this directory. The python scripts load the best config from tuning logs. 

To tune from scratch.
- Uncomment the `ansor_tune` and `autotvm_tune` in each `tune_op_ansor.py` and `tune_op_autotvm.py`, respectively.
- Set `tune = True` in `longformer-eval.py`
