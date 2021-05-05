# Benchmark experiments
sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/replica/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica/neural/' ./bin/runWithLimitGPU.sh

# Large Vent grid search
sbatch --export=CONFIG_PATH='configs/Vent/large-grid/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/large-grid/neural/' ./bin/runWithLimitGPU.sh

# Non-stationarity experiments
sbatch --export=CONFIG_PATH='configs/Vent/replica-full/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-random/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-random/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full-random/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-full-random/neural/' ./bin/runWithLimitGPU.sh

# Fractioning and order experiments
sbatch --export=CONFIG_PATH='configs/Vent/replica-fractions/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-fractions/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-fractions-with-test/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/replica-fractions-with-test/neural/' ./bin/runWithLimitGPU.sh

# Generating the best models from the initial grid search and saving to file
sbatch --export=CONFIG_PATH='configs/GoEmotions/models/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/GoEmotions/models/neural/' ./bin/runWithLimitGPU.sh
sbatch --export=CONFIG_PATH='configs/Vent/models/classic/' ./bin/runWithLimit.sh
sbatch --export=CONFIG_PATH='configs/Vent/models/neural/' ./bin/runWithLimitGPU.sh

# Transfer learning Bert from Vent to GoEmotions
sbatch --export=CONFIG_PATH='configs/GoEmotions/transfer-vent/' ./bin/runWithLimitGPU.sh
