MODEL=decapoda-research/llama-7b-hf
TRAIN_FILE=data/sample_med_qa.jsonl
OUTDIR=outputs/qlora_lora


accelerate launch src/train.py \
--model_name $MODEL \
--train_file $TRAIN_FILE \
--output_dir $OUTDIR \
--per_device_train_batch_size 4 \
--num_train_epochs 3 \
--learning_rate 2e-4
