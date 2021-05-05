CONFIG_PATH="${1:-configs/GoEmotions/classic}"
LINE_TO_RUN="${2:-1}"

#
# List all jsons in CONFIG_PATH
# and run the experiment on the LINE_TO_RUN-th path
#
CONFIG_FILE=$(find "$CONFIG_PATH" | sort | grep ".json" | sed "${LINE_TO_RUN}q;d")
python3 src/run.py --config-path $CONFIG_FILE
