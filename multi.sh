CUDA_VISIBLE_DEVICES=0 python second_main.py | tee weighted.log; CUDA_VISIBLE_DEVICES=0 python second_main.py --config use_iterative | tee iterative.log; CUDA_VISIBLE_DEVICES=0 python second_main.py --config use_straight | tee straight.log;