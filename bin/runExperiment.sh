CONFIG_PATH="${1:-configs/GoEmotions/classic}"
LINE_TO_RUN="${2:-1}"

#
# List all jsons in CONFIG_PATH
# and run the experiment on the LINE_TO_RUN-th path
#
find "$CONFIG_PATH" | sort | grep ".json" \
                    | sed "${LINE_TO_RUN}q;d" \
                    | xargs -L1 python3 src/run.py --config-path
