
#docker run -d --rm --runtime=nvidia --gpus device=0 \
#  -v ~/CPrompt:/workspace -v /mnt/datasets/datasets:/datasets -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/hide:1.0 \
lr=1e-2
python -u main.py --config=./exp/cgqa.json \
    --lr ${lr} \
    --slot_log_name cgqa-slot-10-lr${lr} \
    --only_learn_slot

