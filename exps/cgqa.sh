lr=1e-2
docker run -d --rm --runtime=nvidia --gpus device=0 \
  -v ~/CPrompt:/workspace -v /mnt/datasets/datasets:/datasets -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/root/.cache \
  --shm-size 8G liaoweiduo/hide:1.0 \
python -u main.py --config=./exps/cgqa.json \
    --lr ${lr} \
    --log_name cgqa-50-10-tasks-lr${lr}

#slot_lrs=(1e-4 1e-3 5e-3 1e-2)
#devices=(0 1 2 3)
#for run_id in 0 1 2 3; do
#slot_lr=${slot_lrs[${run_id}]}
#device=${devices[${run_id}]}
## -d (detach) --rm
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CPrompt:/workspace -v /mnt/datasets/datasets:/datasets -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/hide:1.0 \
#python -u main.py --config=./exps/cgqa_slot.json \
#    --lr ${slot_lr} \
#    --slot_log_name cgqa-slot-10-lr${slot_lr} \
#    --only_learn_slot
#done
