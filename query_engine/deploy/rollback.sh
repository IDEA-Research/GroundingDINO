#!/usr/bin/env bash
# rollback.sh — stop medical-query-engine and optionally drop its derived DB.
#
# Usage:
#   bash rollback.sh                          # dry run: print what would happen
#   bash rollback.sh --yes                    # stop the unit
#   bash rollback.sh --drop-db                # dry run incl. drop of medical_monitor_qe
#   bash rollback.sh --drop-db --yes          # stop the unit AND drop medical_monitor_qe
#   bash rollback.sh --drop-db --db medical_monitor_qe_test --yes
#
# Runs as root, or as cluster (uses the NOPASSWD sudoers rule for systemctl stop;
# cluster is in the docker group for `docker exec`).
# The DB name is asserted twice: a bash regex here AND a JS regex inside mongosh.
# Only ^medical_monitor_qe(_test)?$ can ever be dropped; medical_monitor_db cannot.
set -euo pipefail

readonly UNIT=medical-query-engine
readonly CONTAINER=medical-mongodb
readonly QE_DB_RE='^medical_monitor_qe(_test)?$'

die()  { printf '\nABORT: %s\n' "$*" >&2; exit 1; }
info() { printf '    %s\n' "$*"; }

drop=0
yes=0
db=medical_monitor_qe
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drop-db) drop=1 ;;
    --db) [[ $# -ge 2 ]] || die "--db needs a value"; db="$2"; shift ;;
    --yes) yes=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done

# Layer 1: bash assertion on the DB name.
if [[ $drop -eq 1 ]]; then
  [[ "$db" =~ $QE_DB_RE ]] || die "DB name '$db' does not match $QE_DB_RE; refusing"
  [[ "$db" != "medical_monitor_db" ]] || die "refusing to touch the source DB"
fi

if [[ ${EUID} -eq 0 ]]; then
  stop_cmd=(/usr/bin/systemctl stop "$UNIT")
else
  stop_cmd=(sudo -n /usr/bin/systemctl stop "$UNIT")
fi
# Layer 2: the same regex is checked again inside mongosh before dropDatabase().
js="const n='${db}'; if (!/^medical_monitor_qe(_test)?\$/.test(n) || n === 'medical_monitor_db') { throw new Error('refusing to drop ' + n); } const r = db.getSiblingDB(n).dropDatabase(); printjson({dropped: n, result: r});"
drop_cmd=(docker exec "$CONTAINER" mongosh --quiet --eval "$js")

echo "Plan:"
echo "  1. ${stop_cmd[*]}"
if [[ $drop -eq 1 ]]; then
  echo "  2. drop Mongo DB '$db' in container $CONTAINER:"
  echo "       docker exec $CONTAINER mongosh --quiet --eval \"$js\""
else
  echo "  2. (no DB drop; pass --drop-db to drop medical_monitor_qe)"
fi
echo "  Not touched: source DB medical_monitor_db, program dir, unit file (use uninstall.sh)."

if [[ $yes -ne 1 ]]; then
  echo
  echo "Dry run only. Re-run with --yes to execute."
  exit 0
fi

echo
echo "==> stopping $UNIT"
if systemctl is-active --quiet "$UNIT"; then
  "${stop_cmd[@]}" || die "could not stop $UNIT (as non-root this needs the sudoers fragment; else use sudo)"
else
  info "(not active; nothing to stop)"
fi
if systemctl is-active --quiet "$UNIT"; then die "$UNIT still active; not dropping anything"; fi
info "stopped."

if [[ $drop -eq 1 ]]; then
  echo "==> dropping DB '$db'"
  command -v docker >/dev/null || die "docker not found"
  "${drop_cmd[@]}"
  info "done. The service would rebuild it with a full ETL on next start."
fi

echo
echo "Next: to remove the unit entirely: sudo bash $(cd -- "$(dirname -- "$0")" && pwd)/uninstall.sh"
echo "      to start again:              sudo systemctl start $UNIT"
