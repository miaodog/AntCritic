
root=/mnt/fengyao.hjj
cognition_framework_home=${root}/cognition/src/cognition
project_home=/mnt/fengyao.hjj/m2-title-generation
export PATH=${root}/miniconda3/envs/antmmf/bin:$PATH
export PYTHONPATH=${root}/cognition/src:${project_home}:${project_home}/transformers/src/:${root}/miniconda3/envs/antmmf/lib

python ${cognition_framework_home}/run.py \
      --src=${project_home}/services/inference_ops.py \
      --config=${project_home}/services/config.json \
      --port 18848


