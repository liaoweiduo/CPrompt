

#slot_lrs=(1e-4 1e-3 5e-3 1e-2)
#devices=(0 1 2 3)
#for run_id in 0 1 2 3; do
#slot_lr=${slot_lrs[${run_id}]}
#device=${devices[${run_id}]}
## -d (detach) --rm
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CPrompt:/workspace -v /mnt/datasets/datasets:/datasets -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/hide:1.0 \
#python -u main.py --config=./exps/cobj.json \
#    --lr ${slot_lr} \
#    --slot_log_name cobj-slot-10-lr${slot_lr} \
#    --only_learn_slot
#done


lrs=(5e-4 1e-3 5e-3 1e-2)
devices=(0 1 3 4)
for run_id in 0 1 2 3; do
lr=${lrs[${run_id}]}
device=${devices[${run_id}]}
# -d (detach) --rm
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CPrompt:/workspace -v /mnt/datasets/datasets:/datasets -v ~/checkpoints:/checkpoints \
  --shm-size 8G liaoweiduo/hide:1.0 \
python -u main.py --config=./exps/cobj.json \
    --log_name cobj-prompt-lr${lr} \
    --lr ${lr}
done