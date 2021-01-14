# Benchmark experiments
sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/neural/' ./bin/runWithLimitGPU.sh

# Non-stationarity experiments
sbatch --export=CONFIG_PATH='configs/Vent/replica-full/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-random/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-random/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full-random/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full-random/neural/' ./bin/runWithLimitGPU.sh

# Generating the best models and saving to file
sbatch --export=CONFIG_PATH='configs/GoEmotions/models/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/models/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/models/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/models/neural/' ./bin/runWithLimitGPU.sh
