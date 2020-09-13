sbatch --export=CONFIG_PATH='configs/GoEmotions/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/neural/' ./bin/runWithLimitGPU.sh
