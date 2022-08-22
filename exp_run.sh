#cuda101=/gruntdata/DL_dataset/qingpei.gqp/software/cuda-10.1
#export PATH=${cuda101}/bin:${cuda101}/lib64:$PATH
#export LD_LIBRARY_PATH=${cuda101}/lib64:${cuda101}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#CUDA_VISIBLE_DEVICES=0,1 /mnt/fengyao.hjj/miniconda3/envs/antmmf/bin/python /mnt/fengyao.hjj/argument_mining/first_main.py --config use_word
CUDA_VISIBLE_DEVICES=0,1 /mnt/fengyao.hjj/miniconda3/envs/antmmf/bin/python /mnt/fengyao.hjj/argument_mining/second_main.py --config use_gru
#CUDA_VISIBLE_DEVICES=0,1 /mnt/fengyao.hjj/miniconda3/envs/antmmf/bin/python /mnt/fengyao.hjj/argument_mining/second_main.py


#kmcli run --user 328002 --name argument-zy-first-stage-word-3 --no-master \
#--app kmaker --priority low --image "reg.docker.alibaba-inc.com/alipay-alps/pytorch1.4.0-cuda10.0-gcc5.4:v0.1" \
#--worker 'cpu=8,memory=65536,gpu=2,gpu_type=v100' 1 \
#--env NCCL_DEBUG=INFO,NCCL_SOCKET_IFNAME=^lo,docker0,NCCL_IB_GID_INDEX=3,\
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gcc-5.4.0/lib:/usr/local/gcc-5.4.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64,\
#PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH \
#"mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport ant-cognition-qyh47.cn-hangzhou.nas.aliyuncs.com:/multimodal-nas /mnt && bash /mnt/fengyao.hjj/mount.sh && source /mnt/fengyao.hjj/env.sh && bash /mnt/fengyao.hjj/argument_mining/exp_run.sh 2>&1 | tee /mnt/fengyao.hjj/argument_mining/0505_first_stage_use_word_true.log && chown 1365089:100 /mnt/fengyao.hjj/argument_mining -R"

