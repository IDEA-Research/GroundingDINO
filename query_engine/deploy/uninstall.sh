#!/usr/bin/env bash
# uninstall.sh — remove the medical-query-engine systemd unit and sudoers fragment.
#
# Usage:  sudo bash /home/cluster/services/query_engine/deploy/uninstall.sh
#
# Does NOT delete data (Mongo DB medical_monitor_qe) and does NOT delete the program
# directory /home/cluster/services/query_engine; it prints how to do that manually.
# Safe to re-run.
set -euo pipefail

readonly UNIT=medical-query-engine
readonly UNIT_DST=/etc/systemd/system/medical-query-engine.service
readonly SUDOERS_DST=/etc/sudoers.d/cluster-medical-query-engine
readonly TARGET=/home/cluster/services/query_engine

step() { printf '\n==> [%s] %s\n' "$1" "$2"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\nABORT: %s\n' "$*" >&2; exit 1; }

[[ ${EUID} -eq 0 ]] || die "run with sudo: sudo bash $0"

step 1 "Stopping $UNIT"
systemctl stop "$UNIT" 2>/dev/null || info "(not running or not installed)"

step 2 "Disabling $UNIT"
systemctl disable "$UNIT" 2>/dev/null || info "(not enabled or not installed)"

step 3 "Removing unit file $UNIT_DST"
[[ "$UNIT_DST" == "/etc/systemd/system/medical-query-engine.service" ]] || die "unexpected unit path '$UNIT_DST'"
if [[ -e "$UNIT_DST" ]]; then rm -f -- "$UNIT_DST"; info "removed."; else info "(already absent)"; fi

step 4 "Removing sudoers fragment $SUDOERS_DST"
[[ "$SUDOERS_DST" == "/etc/sudoers.d/cluster-medical-query-engine" ]] || die "unexpected sudoers path '$SUDOERS_DST'"
if [[ -e "$SUDOERS_DST" ]]; then rm -f -- "$SUDOERS_DST"; info "removed."; else info "(already absent)"; fi
if visudo -c >/dev/null; then info "visudo -c OK."; else info "WARNING: visudo -c reports an issue elsewhere in sudoers; continuing"; fi

step 5 "systemctl daemon-reload"
systemctl daemon-reload

step 6 "systemctl reset-failed $UNIT"
systemctl reset-failed "$UNIT" 2>/dev/null || info "(nothing to reset)"

step 7 "Done. Data and program directory were kept."
cat <<MANUAL
    Still present (delete manually only if you are sure):
      - Program + venv + .env.qe (contains the API token):
          rm -rf -- $TARGET
      - Derived Mongo DB (never the source DB medical_monitor_db):
          bash $TARGET/deploy/rollback.sh --drop-db            (dry run)
          bash $TARGET/deploy/rollback.sh --drop-db --yes      (really drop)
        (run rollback.sh BEFORE deleting $TARGET, or keep a copy of it)
      - Consumer env entries HELPER_DASHBOARD_QUERY_ENGINE_URL / _TOKEN in the
        helper-dashboard / db-agent env files.
MANUAL
