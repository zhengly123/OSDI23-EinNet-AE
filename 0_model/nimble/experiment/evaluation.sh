TEST_SCRIPT=/home/hsh/test-nimble/experiment/run_inference.py

# bs = 1
python $TEST_SCRIPT infogan --bs 1 --mode nimble
python $TEST_SCRIPT fsrcnn --bs 1 --mode nimble
python $TEST_SCRIPT GCN --bs 1 --mode nimble
python $TEST_SCRIPT csrnet --bs 1 --mode nimble
python $TEST_SCRIPT resnet18 --bs 1 --mode nimble
python $TEST_SCRIPT dcgan --bs 1 --mode nimble

# bs = 16
python $TEST_SCRIPT infogan --bs 16 --mode nimble
python $TEST_SCRIPT fsrcnn --bs 16 --mode nimble
python $TEST_SCRIPT GCN --bs 16 --mode nimble
python $TEST_SCRIPT csrnet --bs 16 --mode nimble
python $TEST_SCRIPT resnet18 --bs 16 --mode nimble
python $TEST_SCRIPT dcgan --bs 16 --mode nimble