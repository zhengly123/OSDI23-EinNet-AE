### how to run Nimble

File `OSDI23-EinNet-AE/0_model/nimble/experiment/evaluation.sh` runs all the experiments for nimble;

`run.sh` can be executed to get the data on the provided server.

If you want to install nimble, follow `install-nimble.sh` and apply `nimble.patch` to its source code. Then run `evaluation.sh` to get the evaluating results.

As shown in the Figure 12 of the paper, Nimble does not support the `longformer` model since out of memory.
