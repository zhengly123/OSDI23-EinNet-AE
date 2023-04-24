export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=0
echo $@
$@ |& tee output.txt
ncu --profile-from-start on --target-processes all $@ |& tee output.profile.txt
ncu --profile-from-start on --target-processes all --metrics dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_write.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum $@ |& tee output.metrics.txt
