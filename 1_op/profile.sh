export NVIDIA_TF32_OVERRIDE=0
echo Profiling $@
/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/cuda-11.0.2-npdlw4kj3xsbaam3gedlxm3umfumpujb/bin/ncu --profile-from-start off --target-processes all --metrics dram__bytes_read.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum $@

# Full collection
# ncu --profile-from-start off --target-processes all --metrics dram__bytes_read.sum,dram__sectors_read.sum,dram__bytes_write.sum,lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum $@ |& tee output.metrics.txt
