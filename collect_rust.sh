#!/bin/bash

# Usage: ./concat_rust.sh [directory] [output_file]
DIR="${1:-.}"
OUTPUT="${2:-combined.rs}"

> "$OUTPUT"

find "$DIR" -name "*.rs" | sort | while read -r file; do
  echo "// ============================================================" >> "$OUTPUT"
  echo "// FILE: $file" >> "$OUTPUT"
  echo "// ============================================================" >> "$OUTPUT"
  echo "" >> "$OUTPUT"
  cat "$file" >> "$OUTPUT"
  echo "" >> "$OUTPUT"
  echo "" >> "$OUTPUT"
done

echo "Done! Combined into: $OUTPUT"
