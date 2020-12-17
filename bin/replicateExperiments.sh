sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/neural/' ./bin/runWithLimitGPU.sh
