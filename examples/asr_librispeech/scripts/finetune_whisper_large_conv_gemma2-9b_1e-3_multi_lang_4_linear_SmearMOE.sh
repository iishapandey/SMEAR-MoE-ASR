#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# bash /raid/isha/SpeechLLM/SLAM-LLM/examples/asr_librispeech/scripts/finetune_Indicwhisper_large_conv_gemma2-9b_1e-3.sh
# source /raid/isha/SpeechLLM/SLAM-LLM/slam_venv/bin/activate
# screen -S finetune_gemma_indicwhisper_conv_1e-3
date_str=$(date +"%Y%m%d")
run_dir=/raid/isha/SpeechLLM/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/raid/isha/SpeechLLM/SLAM-LLM/models_pretrained/audio_encoder/whisper-large-v3.pt

llm_path=/raid/isha/SpeechLLM/SLAM-LLM/models_pretrained/LLMs/gemma2-9b


# to be changed
train_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/multilingual/train_hi_mr_te_ta_indicsuperb_new_indicvoicesV3_merged_filtered_wer_30_prompt_corrected.jsonl
val_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/multilingual/val_hi_mr_te_ta_indicsuperb_indicvoicesV3_merged_filtered.jsonl
# train_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/multilingual/val_hi_mr_te_ta_indicsuperb_indicvoicesV3_merged_filtered.jsonl
# train_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_kathbath_prompt_corrected.jsonl
# val_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_kathbath_prompt_corrected.jsonl
# train_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/multilingual/val_hi_mr_te_ta_indicsuperb_indicvoicesV3_merged_filtered.jsonl
# val_data_path=/raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_kathbath_prompt_corrected.jsonl
# /raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_commonvoice_prompt_corrected.jsonl
# /raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_fleurs_prompt_corrected.jsonl
# /raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_indictts_prompt_corrected.jsonl
# /raid/isha/SpeechLLM/SLAM-LLM/datasets/Malayalam/test_malayalam_Vistar_kathbath_prompt_corrected.jsonl

output_dir=/raid/isha/SpeechLLM/SLAM-LLM/outputs/gemma2-9b-fp32-IndicSuperb-IndicVoicesv3_new-cov1d-linear-steplrwarmupkeep1e-3-whisper-multi_lang_4_linear_SmearMoe_3_experts_$date_str
# ckpt_dir=/raid/isha/SpeechLLM/SLAM-LLM/outputs/gemma2-9b-fp32-IndicSuperb-IndicVoicesv3_new-cov1d-linear-steplrwarmupkeep1e-3-whisper-multi_lang_4_linear_SmearMoe_20250907/asr_epoch_1_step_1000/model.pt

hydra_args="\
hydra.run.dir=$output_dir \
++model_config.llm_name=gemma2-9b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=3584 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf="openai/whisper-large-v3"
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=ds-conv-linear \
++model_config.moe_gating=true \
++model_config.moe_routing="smear_utter" \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.file="src/slam_llm/datasets/speech_dataset_multi_lang.py:get_speech_dataset" \
++train_config.model_name=asr \
++train_config.num_epochs=15 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=10000 \
++train_config.total_steps=200000 \
++train_config.lr=1e-3 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=7 \
++train_config.val_batch_size=7 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++train_config.save_on_best_val_loss=False \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir/wandb \
++log_config.wandb_entity_name=iishapandey77 \
++log_config.wandb_project_name=SLAM_ASR \
++log_config.wandb_exp_name=gemma2-9b-fp32-IndicSuperb-IndicVoicesv3_new-cov1d-linear-steplrwarmupkeep1e-3-whisper-multi_lang_4_linear_SmearMoe_3_experts_$date_str \
++log_config.log_interval=100 \
++metric=acc \
++num_lang=3 \
"
# ++ckpt_path=$ckpt_dir \


# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5779 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 8\
        --master_port=22575 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        $hydra_args
fi