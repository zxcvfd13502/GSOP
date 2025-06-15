export HF_HOME='Your HF Home Path'

python3 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_gqa \
    --attn_implementation eager\
    --output_path ./logs/check/\
    --search_config_path ./search_cfgs/lv157.yaml \
    --mid_out_dir ./search_res/cls_att_g12o120_gqa_500q_llh_s1_step20\
    --greedy_exp_name r025 \
    --acc_threshold_ratio 0.3 --greedy_steps 20\
    --greedy_task 'gqa' --greedy_metric 'exact_match,none' \
    --img_token_grouping 'cls' --key_img_ratio 0.25 \
    --prune_groups 1 2 --prune_ops 1 2 0 \
    --time_limit 2.0 --limit 500 --ll_half  --remove_key_layer 6 6 6