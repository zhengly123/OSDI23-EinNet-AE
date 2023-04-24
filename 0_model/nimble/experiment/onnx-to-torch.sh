INPUTDIR="/home/hsh/test-nimble/models"
OUTPUTDIR="/home/hsh/test-nimble/pytorch_model"

for file in ${INPUTDIR}/*.onnx
do
    echo $file
    python /home/hsh/test-nimble/experiment/onnx_to_torch.py $file ${OUTPUTDIR}/$(basename $file).pth
done
