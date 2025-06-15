export HF_HOME='/projectnb/ivc-ml/amliu/huggingface'

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port 29811 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks gqa,mmmu_val,mme,pope,ok_vqa_val2014,mmbench_en_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix gsop_org_035 \
    --attn_implementation eager\
    --output_path ./logs/check/\
    --desire_prune_ratio 0.30\
    --prune_config_path /projectnb/ivc-ml/amliu/slm/lmms-eval/open_v1_res/clsatt/img_only/lv157/gqa/r1_4/cls_att_g12o120_gqa_500q_llh_s1_step15