#!/bin/bash

# Name of the Conda environment
ENV_NAME="Diff-MI"

# Arrays of package names, target filenames, line numbers, and new line contents
PACKAGE_NAMES=("robustness" "pytorch_fid")
TARGET_FILENAMES=("attacker.py" "fid_score.py")
LINE_NUMBERS=(156 242)
NEW_LINE_CONTENTS=(
    "            output = self.model(inp)[-1]" 
    "                       for file in path.rglob('*.{}'.format(ext))])"
)

# Activate the Conda environment
source activate $ENV_NAME

# Loop through each package
for i in "${!PACKAGE_NAMES[@]}"; do
    PACKAGE_NAME="${PACKAGE_NAMES[$i]}"
    TARGET_FILENAME="${TARGET_FILENAMES[$i]}"
    LINE_NUMBER="${LINE_NUMBERS[$i]}"
    NEW_LINE_CONTENT="${NEW_LINE_CONTENTS[$i]}"
    echo $NEW_LINE_CONTENT

    # Get the path of the package in the Conda environment
    PACKAGE_PATH=$(python -c "import os, $PACKAGE_NAME; print(os.path.dirname($PACKAGE_NAME.__file__))")

    # Find the file within the package directory
    FILE_PATH=$(find "$PACKAGE_PATH" -name "$TARGET_FILENAME")

    # Check if the file exists
    if [ -f "$FILE_PATH" ]; then
        # Use sed to modify the specific line
        sed -i "${LINE_NUMBER}s/.*/$NEW_LINE_CONTENT/" "$FILE_PATH"
        echo "Modified $FILE_PATH"
    else
        echo "File $FILE_PATH not found!"
    fi
done
