#!/usr/bin/env bash
# Scan every *.log file in this directory for ERROR-level entries and
# print a per-file summary along with an aggregated message-frequency table.
#
# Usage:  ./analyze_errors.sh [log-dir]
# Default log-dir is the directory the script lives in.

set -euo pipefail

dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ ! -d "$dir" ]]; then
    echo "directory not found: $dir" >&2
    exit 1
fi

shopt -s nullglob
files=("$dir"/*.log)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
    echo "no .log files in $dir"
    exit 0
fi

total=0
declare -a per_file_counts=()

printf '== Per-file ERROR summary ==\n'
for f in "${files[@]}"; do
    name=$(basename "$f")
    # Match the log-level token "ERROR" (env_logger style: "<ts> ERROR  [module] msg").
    # -n keeps line numbers so the user can jump directly into the file.
    matches=$(grep -nE '^\S+\s+ERROR\b' "$f" || true)
    if [[ -z "$matches" ]]; then
        per_file_counts+=("$name 0")
        continue
    fi
    count=$(printf '%s\n' "$matches" | wc -l | tr -d ' ')
    per_file_counts+=("$name $count")
    total=$((total + count))

    printf '\n--- %s (%s errors) ---\n' "$name" "$count"
    # Show "line:  [module] message" — strip the timestamp+level prefix for readability.
    printf '%s\n' "$matches" | awk -F: '{
        ln=$1; $1=""; sub(/^:/, "");
        # collapse leading "<ts> ERROR  " from the remainder
        sub(/^[[:space:]]*[^ ]+[[:space:]]+ERROR[[:space:]]+/, "");
        printf "  L%s  %s\n", ln, $0
    }'
done

printf '\n== Counts by file ==\n'
printf '%s\n' "${per_file_counts[@]}" | sort -k2 -n -r | awk '{ printf "  %-20s %s\n", $1, $2 }'

printf '\n== Top error messages (across all files) ==\n'
if [[ $total -eq 0 ]]; then
    printf '  (no ERROR-level entries found)\n'
else
    # Strip timestamp + level + module prefix to group similar messages.
    grep -hE '^\S+\s+ERROR\b' "${files[@]}" \
        | sed -E 's/^\S+\s+ERROR\s+\[[^]]+\]\s+//' \
        | sort | uniq -c | sort -rn | head -20 \
        | awk '{ count=$1; $1=""; sub(/^ /, ""); printf "  %5d  %s\n", count, $0 }'
fi

printf '\nTotal ERROR entries: %d\n' "$total"
