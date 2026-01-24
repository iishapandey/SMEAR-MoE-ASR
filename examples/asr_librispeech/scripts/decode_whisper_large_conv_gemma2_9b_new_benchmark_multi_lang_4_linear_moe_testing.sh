#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
# source <path to smear-moe-asr dir>/slam_venv/bin/activate
run_dir=<path to smear-moe-asr dir>
cd $run_dir
code_dir=examples/asr_librispeech
op_dir=<path to smear-moe-asr dir>/outputs
speech_encoder_path=<path to smear-moe-asr dir>/models_pretrained/audio_encoder/whisper-large-v3.pt
llm_path=<path to smear-moe-asr dir>/models_pretrained/LLMs/gemma2-9b

lang=${LANG}  # options: "hi", "mr", "ta", "te", "test"

# # ------------------------------------------------------------------------------------------------------------------------------------------
# ckpt_dir="${op_dir}/gemma2-9b-fp32-IndicSuperb-IndicVoicesv3_new-cov1d-linear-steplrwarmupkeep1e-3-whisper-multi_lang_4_linear_SmearMoe_20250902"
# ckpt_path="${ckpt_dir}/asr_epoch_5_step_7772"


declare -A datasets
declare -A splitkeys

datasets[test]="
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_100_sampled.jsonl
"

splitkeys[test]="test_val_hin_30"


datasets[hi]="
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_commonvoice_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_fleurs_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_indictts_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_kathbath_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/hindi/test_hindi_Vistar_mucs_prompt_corrected.jsonl
"


splitkeys[hi]="commonvoice_val_hi_30 fleurs_val_hi_30 indictts_val_hi_30 kathbath_val_hi_30 mucs_val_hi_30"

datasets[mr]="
<path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_commonvoice_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_fleurs_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_indictts_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_kathbath_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_mucs_prompt_corrected.jsonl
"

# <path to smear-moe-asr dir>/datasets/Marathi/test_marathi_Vistar_indicvoices_prompt_corrected.jsonl
 
splitkeys[mr]="commonvoice_val_mr_30 fleurs_val_mr_30 indictts_val_mr_30 kathbath_val_mr_30 mucs_val_mr_30"


datasets[ta]="
<path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_commonvoice_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_fleurs_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_indictts_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_kathbath_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_mucs_prompt_corrected.jsonl
"
# <path to smear-moe-asr dir>/datasets/Tamil/test_tamil_Vistar_indicvoices_prompt_corrected.jsonl

splitkeys[ta]="commonvoice_val_ta_30 fleurs_val_ta_30 indictts_val_ta_30 kathbath_val_ta_30 mucs_val_ta_30"

datasets[te]="
<path to smear-moe-asr dir>/datasets/Telugu/test_telugu_Vistar_fleurs_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Telugu/test_telugu_Vistar_indictts_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Telugu/test_telugu_Vistar_kathbath_prompt_corrected.jsonl
<path to smear-moe-asr dir>/datasets/Telugu/test_telugu_Vistar_mucs_prompt_corrected.jsonl
"

# <path to smear-moe-asr dir>/datasets/Telugu/test_telugu_Vistar_indicvoices_prompt_corrected.jsonl

splitkeys[te]="fleurs_val_te_30 indictts_val_te_30 kathbath_val_te_30 mucs_val_te_30"


# Convert to arrays
IFS=$'\n' read -r -d '' -a val_data_paths < <(printf "%s\0" "${datasets[$lang]}")
IFS=' ' read -r -a splits <<< "${splitkeys[$lang]}"

# Debug: Print val_data_paths
echo "=== val_data_paths for language [$lang] ==="
for path in "${val_data_paths[@]}"; do
    echo "$path"
done

# Debug: Print splits
echo "=== splits for language [$lang] ==="
for split in "${splits[@]}"; do
    echo "$split"
done


# Loop through each val_data_path and split value
for i in "${!val_data_paths[@]}"; do
    val_data_path="${val_data_paths[$i]}"
    split="${splits[$i]}"
    # ckpt_dir="<path to smear-moe-asr dir>/outputs/nemo-4b-IndicVoices-linear-steplrwarmupkeep1e-4-whisper-largev3-20241226/"
    # ckpt_path="${ckpt_dir}/asr_epoch_4_step_1736"
    
    lang_ckpt_path="${ckpt_path}/${lang}"
    mkdir -p "$lang_ckpt_path"
    output_dir="${lang_ckpt_path}/output-${split}"
    decode_log="${lang_ckpt_path}/decode_${split}_beam4"

    echo "Running for val_data_path: $val_data_path and split: $split"

    python $code_dir/inference_asr_batch.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            hydra.run.dir=$ckpt_path \
            ++model_config.llm_name="gemma2-9b" \
            ++model_config.llm_path=$llm_path \
            ++model_config.llm_dim=3584 \
            ++model_config.encoder_name=whisper \
            ++model_config.encoder_projector_ds_rate=5 \
            ++model_config.encoder_path=$speech_encoder_path \
            ++model_config.encoder_path_hf="openai/whisper-large-v3" \
            ++model_config.encoder_dim=1280 \
            ++model_config.encoder_projector=ds-conv-linear \
            ++model_config.moe_gating=true \
            ++model_config.moe_routing="smear_utter" \
            ++dataset_config.dataset=speech_dataset \
            ++dataset_config.val_data_path=$val_data_path \
            ++dataset_config.input_type=mel \
            ++dataset_config.mel_size=128 \
            ++dataset_config.file="src/slam_llm/datasets/speech_dataset_multi_lang.py:get_speech_dataset" \
            ++dataset_config.inference_mode=true \
            ++train_config.model_name=asr \
            ++train_config.freeze_encoder=true \
            ++train_config.freeze_llm=true \
            ++train_config.batching_strategy=custom \
            ++train_config.num_epochs=1 \
            ++train_config.val_batch_size=30 \
            ++train_config.num_workers_dataloader=2 \
            ++train_config.output_dir=$output_dir \
            ++decode_log=$decode_log \
            ++ckpt_path=$ckpt_path/model.pt \
            ++num_lang=4 \
            # ++peft_ckpt=$ckpt_path \
            # ++train_config.use_peft=true \
            # ++train_config.peft_config.r=32 \
            # ++dataset_config.normalize=true \
            # ++model_config.encoder_projector=q-former \
            # ++dataset_config.fix_length_audio=64 \
done