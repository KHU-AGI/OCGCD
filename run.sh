
conda --version
python --version
SEEDS="1 3 8 2 4"
for Seed in $SEEDS; do
    echo "[Experimental Seed: $Seed]"
    python Ours_train_online.py --model VIT --dataset cub --alpha 32 --mrg 0.1 --lr 1e-4 --warm 2 --base_epochs 30 --inc_epochs 15 --batch-size 64 --inc_batch-size 64 --num_base_classes 160 --num_inc_sessions 1 --seed $Seed \
    --exp "DEAN_Repro" --lora_layers 7 8 9 10 11 --energy_hp 1. --n_replay 5 --use_EC --use_VFA
done
