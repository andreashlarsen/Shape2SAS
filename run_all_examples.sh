#!/bin/bash

README="README.md"
TMP_SCRIPT="run_extracted_examples.sh"

echo "#!/bin/bash" > $TMP_SCRIPT
echo "" >> $TMP_SCRIPT

# Flags
inside_code_block=0

while IFS= read -r line; do
    # Toggle code block flag
    if [[ "$line" == '```'* ]]; then
        inside_code_block=$((1 - inside_code_block))
        continue
    fi

    # If heading (### Example …), add echo separators
    if [[ "$line" =~ ^###\ Example ]]; then
        echo "echo --------------------------------------------------------------------" >> $TMP_SCRIPT
        # Strip leading ### and trim spaces
        heading=$(echo "$line" | sed 's/^### //' | sed 's/^ *//')
        echo "echo $heading" >> $TMP_SCRIPT
        echo "echo --------------------------------------------------------------------" >> $TMP_SCRIPT
        continue
    fi

    # If we are inside a code block, capture python/open commands
    if [[ $inside_code_block -eq 1 ]]; then
        if [[ "$line" =~ ^python|^open ]]; then
            echo "echo Running: $line" >> $TMP_SCRIPT
            echo "$line" >> $TMP_SCRIPT
            echo "" >> $TMP_SCRIPT
        fi
    fi
done < "$README"

# Remove Windows carriage returns just in case
tr -d '\r' < $TMP_SCRIPT > "${TMP_SCRIPT}.tmp" && mv "${TMP_SCRIPT}.tmp" $TMP_SCRIPT

# Make executable
chmod +x $TMP_SCRIPT
echo "Extracted script written to $TMP_SCRIPT"

# Run it
echo "Running $TMP_SCRIPT ..."
./$TMP_SCRIPT

