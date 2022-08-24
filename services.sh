
cognition_framework_home=/mnt/fengyao.hjj/cognition/src/cognition
project_home=/mnt/fengyao.hjj/argument_mining
export PATH=/root/miniconda3/envs/antcritic/bin:$PATH
export PYTHONPATH=/mnt/fengyao.hjj/cognition/src:${project_home}:/root/miniconda3/envs/antcritic/lib

python ${cognition_framework_home}/run.py \
      --src=${project_home}/services/inference_ops.py \
      --config=${project_home}/services/config.json \
      --port 18848


