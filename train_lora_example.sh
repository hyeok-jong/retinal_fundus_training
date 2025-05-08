




start_time=$SECONDS
model_name_s=('dinov2' 'mae' 'openclip')
learning_rate_s=(0.0001)
optimizer_s=('adamw')
weight_decay_s=(0.001)
learning_method_s=('lora')


for model_name in "${model_name_s[@]}"; do
    for learning_rate in "${learning_rate_s[@]}"; do
        for learning_method in "${learning_method_s[@]}"; do
            for weight_decay in "${weight_decay_s[@]}"; do
                for optimizer in "${optimizer_s[@]}"; do

                    current_time=$(date -u -d '9 hours' +%T)
                    echo "Current time in KST: $current_time"
                    elapsed_time=$((SECONDS - start_time))
                    hours=$((elapsed_time / 3600))
                    minutes=$(( (elapsed_time % 3600) / 60))
                    seconds=$((elapsed_time % 60))

                    echo "Current Time: $current_time"
                    echo "Elapsed Time: $hours hour(s) $minutes minute(s) $seconds second(s)"

                    CUDA_VISIBLE_DEVICES=1,2,3,4 \
                    python main.py \
                    --model_name "${model_name}" \
                    --learning_method "${learning_method}" \
                    --lr "${learning_rate}" \
                    --epochs 100 \
                    --optimizer "${optimizer}" \
                    --weight_decay "${weight_decay}" \
                    --scheduler CAWR \
                    --batch_size 8 \
                    --image_size 224 \
                    --mixup no \
                    --agesex yes \
                    --master_port 12355
                done
            done
        done
    done
done




