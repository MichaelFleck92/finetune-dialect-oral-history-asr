#!/bin/bash
# Arbeitet training/jobs/queue.txt ab: je Zeile ein Befehl, die oberste zuerst.
# Die Datei darf waehrend des Laufs geaendert werden (Zeilen ergaenzen, umstellen, loeschen).
# Start im Hintergrund:  nohup bash training/jobs/queue.sh >> training/jobs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
export PYTHONUNBUFFERED=1
Q=training/jobs/queue.txt
while true; do
  line=$(grep -v '^\s*\(#\|$\)' "$Q" 2>/dev/null | head -n 1)
  [ -z "$line" ] && { echo "$(date '+%F %T') QUEUE EMPTY"; break; }
  grep -v -x -F -- "$line" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"     # Zeile entnehmen
  echo "$(date '+%F %T') START $line"
  bash -c "$line"
  echo "$(date '+%F %T') EXIT $? $line"
  echo "$line" >> training/jobs/done.txt
done
