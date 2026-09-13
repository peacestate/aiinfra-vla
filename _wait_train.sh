#!/usr/bin/env bash
# Lightweight watcher: curl only, no Python per tick.
T=$(tr -d ' \r\n' < "$HOME/.kaggle/access_token")
URL="https://www.kaggle.com/api/v1/kernels/status?userName=kianukal&kernelSlug=langact-train"
for i in $(seq 1 60); do
  R=$(curl -s -m 30 -H "Authorization: Bearer $T" -H "User-Agent: hyperframe/1.0" "$URL")
  case "$R" in
    *running*|*RUNNING*) : ;;
    *complete*|*COMPLETE*|*error*|*ERROR*|*cancel*|*CANCEL*)
        echo "FINISHED $(date +%H:%M:%S): $R"; exit 0;;
    *) echo "$(date +%H:%M:%S) odd: ${R:0:120}";;
  esac
  sleep 150
done
echo "TIMEOUT after 2.5h"; exit 1
