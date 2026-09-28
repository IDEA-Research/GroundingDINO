#!/usr/bin/env bash
# install.sh — install / upgrade medical-query-engine (nicu_qe) on orin11.
#
# Usage (run by the USER, after reading this file):
#   less ~/qe-stage/query_engine/deploy/install.sh
#   sudo bash ~/qe-stage/query_engine/deploy/install.sh ~/qe-stage/query_engine
#
# Safe to re-run:
#   - never overwrites an existing .env.qe (token is kept)
#   - never kills any process; aborts if the port is held by anything other than
#     this unit's own main process
#   - re-running on a running service restarts it with the new code
# Stops at the first failure (set -euo pipefail).
set -euo pipefail

# ---- fixed paths (asserted before any delete) ------------------------------
readonly SVC_USER=cluster
readonly SERVICES_DIR=/home/cluster/services
readonly TARGET=/home/cluster/services/query_engine
readonly VENV=/home/cluster/services/query_engine/.venv-qe
readonly ENV_FILE=/home/cluster/services/query_engine/.env.qe
readonly BASE_PY=/home/cluster/miniforge3/bin/python3.12
readonly UNIT=medical-query-engine
readonly UNIT_DST=/etc/systemd/system/medical-query-engine.service
readonly SUDOERS_DST=/etc/sudoers.d/cluster-medical-query-engine
readonly QE_DB_RE='^medical_monitor_qe(_test)?$'

step() { printf '\n==> [%s] %s\n' "$1" "$2"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die()  { printf '\nABORT: %s\n' "$*" >&2; exit 1; }
as_user() { sudo -u "$SVC_USER" -H -- "$@"; }

# ---- 0. preconditions -------------------------------------------------------
step 0 "Checking preconditions"
[[ ${EUID} -eq 0 ]] || die "run with sudo: sudo bash $0 <source_dir>"
[[ "${SUDO_USER:-}" == "$SVC_USER" ]] || die "run as '$SVC_USER' via sudo (SUDO_USER='${SUDO_USER:-}')"
[[ $# -eq 1 ]] || die "usage: sudo bash $0 <source_dir>   (the directory containing nicu_qe/)"
SRC="$(cd -- "$1" && pwd -P)" || die "source_dir '$1' not found"
[[ -f "$SRC/nicu_qe/__init__.py" ]] || die "$SRC/nicu_qe/__init__.py missing: not a query_engine source dir"
[[ -f "$SRC/pyproject.toml" ]] || die "$SRC/pyproject.toml missing"
for f in medical-query-engine.service sudoers-cluster-medical-query-engine env.qe.example; do
  [[ -f "$SRC/deploy/$f" ]] || die "$SRC/deploy/$f missing"
done
case "$SRC/" in
  "$TARGET"/*) die "source_dir must not be (inside) $TARGET; stage it elsewhere, e.g. ~/qe-stage/query_engine" ;;
esac
[[ -x "$BASE_PY" ]] || die "$BASE_PY not found"
for cmd in rsync ss curl systemctl visudo install sudo; do
  command -v "$cmd" >/dev/null || die "required command '$cmd' not found"
done
info "source : $SRC"
info "target : $TARGET"

# ---- 1. sync code (as cluster) ---------------------------------------------
step 1 "Syncing code to $TARGET (keeps .venv-qe and .env.qe)"
install -d -o "$SVC_USER" -g "$SVC_USER" -m 755 "$SERVICES_DIR" "$TARGET"
# rsync --delete only ever targets this exact directory:
[[ "$TARGET" == "/home/cluster/services/query_engine" ]] || die "unexpected TARGET '$TARGET'"
# Excluded paths on the receiver are protected from --delete (no --delete-excluded),
# so .venv-qe and .env.qe survive.
as_user rsync -a --delete \
  --exclude '.venv*' --exclude '*.egg-info' --exclude .pytest_cache \
  --exclude __pycache__ --exclude .env.qe --exclude '*.pid' --exclude .git \
  "$SRC"/ "$TARGET"/
info "done."

# ---- 2. venv (as cluster) ---------------------------------------------------
step 2 "Python venv $VENV"
import_ok() {
  as_user "$VENV/bin/python" -c 'import nicu_qe, pymongo, fastapi, uvicorn, pydantic' >/dev/null 2>&1
}
remove_venv() {
  [[ "$VENV" == "/home/cluster/services/query_engine/.venv-qe" ]] || die "refusing to delete unexpected venv path '$VENV'"
  rm -rf -- "$VENV"
}
if [[ -x "$VENV/bin/python" ]] && import_ok && [[ "${QE_FORCE_PIP:-0}" != 1 ]]; then
  info "venv exists and imports OK; skipping pip (run with QE_FORCE_PIP=1 to reinstall)."
else
  if [[ ! -x "$VENV/bin/python" ]]; then
    info "creating venv with $BASE_PY"
    as_user "$BASE_PY" -m venv "$VENV"
  fi
  info "pip install -e $TARGET (needs network for dependencies)"
  if as_user "$VENV/bin/python" -m pip install --disable-pip-version-check -e "$TARGET"; then
    info "pip install OK."
  else
    warn "pip install failed. Rebuilding venv with --system-site-packages"
    warn "(uses the miniforge base packages: fastapi / uvicorn / pydantic / pymongo)."
    remove_venv
    as_user "$BASE_PY" -m venv --system-site-packages "$VENV"
    if as_user "$VENV/bin/python" -m pip install --disable-pip-version-check \
         --no-deps --no-build-isolation -e "$TARGET"; then
      info "editable install (no deps) OK."
    else
      warn "editable install failed too; adding a .pth path entry for $TARGET instead."
      site="$(as_user "$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
      [[ "$site" == "$VENV"/* ]] || die "unexpected site-packages path '$site'"
      printf '%s\n' "$TARGET" | as_user tee "$site/nicu_qe_local.pth" >/dev/null
    fi
  fi
  import_ok || die "venv cannot import nicu_qe/pymongo/fastapi/uvicorn/pydantic; see messages above"
fi
as_user "$VENV/bin/python" -c 'import sys, pymongo, fastapi, pydantic, nicu_qe; print("    python", sys.version.split()[0], "| pymongo", pymongo.version, "| fastapi", fastapi.__version__, "| pydantic", pydantic.VERSION)'

# ---- 3. .env.qe -------------------------------------------------------------
step 3 "Environment file $ENV_FILE"
if [[ -e "$ENV_FILE" ]]; then
  info "exists; NOT modified (token kept)."
else
  info "not found; creating from deploy/env.qe.example with a new QE_API_TOKEN"
  if command -v openssl >/dev/null; then
    token="$(openssl rand -hex 32)"
  else
    token="$("$BASE_PY" -c 'import secrets; print(secrets.token_hex(32))')"
  fi
  [[ "$token" =~ ^[0-9a-f]{64}$ ]] || die "token generation failed"
  tmp="$(mktemp)"
  sed "s/^QE_API_TOKEN=.*/QE_API_TOKEN=${token}/" "$SRC/deploy/env.qe.example" > "$tmp"
  install -o "$SVC_USER" -g "$SVC_USER" -m 600 "$tmp" "$ENV_FILE"
  rm -f -- "$tmp"
  unset token
  info "created (mode 600, owner $SVC_USER)."
fi
chown "$SVC_USER:$SVC_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

# Safety checks on the effective values (never prints the token).
env_get() { grep -E "^$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true; }
qe_db="$(env_get QE_DB)";         qe_db="${qe_db:-medical_monitor_qe}"
src_db="$(env_get QE_SOURCE_DB)"; src_db="${src_db:-medical_monitor_db}"
bind="$(env_get QE_BIND)";        bind="${bind:-127.0.0.1:8020}"
[[ "$qe_db" =~ $QE_DB_RE ]] || die "QE_DB='$qe_db' in $ENV_FILE does not match $QE_DB_RE"
[[ "$qe_db" != "$src_db" ]] || die "QE_DB equals QE_SOURCE_DB ('$qe_db'); refusing"
[[ -n "$(env_get QE_API_TOKEN)" ]] || warn "QE_API_TOKEN is empty; service will refuse to start unless QE_ALLOW_NO_TOKEN=1"
port="${bind##*:}"
host="${bind%:*}"
[[ "$port" =~ ^[0-9]+$ ]] || die "cannot parse port from QE_BIND='$bind'"
case "$host" in
  127.0.0.1|localhost) ;;
  *) warn "QE_BIND host '$host' is not 127.0.0.1/localhost" ;;
esac
info "QE_DB=$qe_db  QE_SOURCE_DB=$src_db  QE_BIND=$bind"

# ---- 4. systemd unit --------------------------------------------------------
step 4 "Installing unit $UNIT_DST"
install -o root -g root -m 644 "$SRC/deploy/medical-query-engine.service" "$UNIT_DST"
if command -v systemd-analyze >/dev/null; then
  systemd-analyze verify "$UNIT_DST" || warn "systemd-analyze verify reported issues (see above); continuing"
fi

# ---- 5. sudoers fragment ----------------------------------------------------
step 5 "Installing sudoers fragment $SUDOERS_DST"
tmp="$(mktemp)"
cp -- "$SRC/deploy/sudoers-cluster-medical-query-engine" "$tmp"
if ! visudo -cf "$tmp"; then
  rm -f -- "$tmp"
  die "sudoers fragment failed visudo -cf; nothing installed"
fi
install -o root -g root -m 440 "$tmp" "$SUDOERS_DST"
rm -f -- "$tmp"
if ! visudo -c >/dev/null; then
  [[ "$SUDOERS_DST" == "/etc/sudoers.d/cluster-medical-query-engine" ]] || die "unexpected sudoers path"
  rm -f -- "$SUDOERS_DST"
  die "full sudoers check failed after install; removed $SUDOERS_DST"
fi
info "visudo OK."

# ---- 6. daemon-reload -------------------------------------------------------
step 6 "systemctl daemon-reload"
systemctl daemon-reload

# ---- 7. port check (never kills anything) -----------------------------------
step 7 "Checking port $port"
was_active=0
if systemctl is-active --quiet "$UNIT"; then was_active=1; fi
# No `ss | grep -q` pipeline: under pipefail an early grep exit can SIGPIPE ss and flip the result.
listeners="$(ss -ltnp 2>/dev/null || true)"
if grep -qF ":$port " <<<"$listeners"; then
  holders="$(ss -ltnpH "sport = :$port" 2>/dev/null || true)"
  main_pid="$(systemctl show -p MainPID --value "$UNIT" 2>/dev/null || echo 0)"
  pids="$(printf '%s\n' "$holders" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
  foreign=""
  for p in $pids; do
    if [[ "$p" != "$main_pid" ]]; then foreign="$foreign $p"; fi
  done
  if [[ $was_active -eq 1 && -n "$pids" && -z "$foreign" ]]; then
    info "port $port is held by $UNIT itself (MainPID $main_pid); it will be restarted."
  else
    printf '%s\n' "$holders" >&2
    for p in $pids; do ps -o pid,user,etime,cmd -p "$p" >&2 || true; done
    die "port $port is in use by another process (listed above). Nothing was killed. Free the port or change QE_BIND, then re-run."
  fi
else
  info "port $port is free."
fi

# ---- 8. enable + start ------------------------------------------------------
step 8 "Enabling and starting $UNIT"
systemctl reset-failed "$UNIT" 2>/dev/null || true
systemctl enable "$UNIT"
if [[ $was_active -eq 1 ]]; then
  systemctl restart "$UNIT"
else
  systemctl start "$UNIT"
fi

# ---- 9. health --------------------------------------------------------------
step 9 "Waiting 10 s, then GET http://127.0.0.1:$port/health"
sleep 10
ok=0
for attempt in 1 2 3; do
  if body="$(curl -fsS --max-time 5 "http://127.0.0.1:$port/health")"; then
    printf '%s\n' "$body"
    ok=1
    break
  fi
  info "attempt $attempt failed; retrying in 5 s"
  sleep 5
done
restarts="$(systemctl show -p NRestarts --value "$UNIT")"
info "systemctl show -p NRestarts: NRestarts=$restarts   ActiveState=$(systemctl show -p ActiveState --value "$UNIT")"
if [[ $ok -ne 1 ]]; then
  systemctl --no-pager status "$UNIT" || true
  journalctl -u "$UNIT" -n 40 --no-pager || true
  curl -sS --max-time 5 "http://127.0.0.1:$port/health" || true
  die "/health did not return 200. HTTP 503 = Mongo unreachable (check: docker ps | grep medical-mongodb). See RUNBOOK.md section 7. To stop: sudo systemctl stop $UNIT"
fi
[[ "$restarts" == 0 ]] || warn "NRestarts=$restarts (expected 0); check journalctl -u $UNIT"

# ---- 10. next steps ---------------------------------------------------------
step 10 "Installed. Next steps"
cat <<NEXT
    1. Give the token to the consumers (helper-dashboard / db-agent). Read it with
         grep '^QE_API_TOKEN=' $ENV_FILE
       and put into the consumer's env file:
         HELPER_DASHBOARD_QUERY_ENGINE_URL=http://127.0.0.1:$port
         HELPER_DASHBOARD_QUERY_ENGINE_TOKEN=<that value>
       then restart the consumer.
    2. Verify indexes:  cd $TARGET && (set -a; . ./.env.qe; set +a; .venv-qe/bin/python -m nicu_qe indexes check)
    3. Watch ETL:       curl -s http://127.0.0.1:$port/health   (etl.last_ok, etl.lag_seconds)
    4. Logs:            journalctl -u $UNIT -f
    Rollback:  bash $TARGET/deploy/rollback.sh          (dry run; add --yes to act)
    Uninstall: sudo bash $TARGET/deploy/uninstall.sh
NEXT
