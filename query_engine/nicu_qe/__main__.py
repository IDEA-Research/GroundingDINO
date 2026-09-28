"""CLI: python -m nicu_qe serve | etl --once|--full|--loop|--reconcile | indexes ensure|check|drop | check.

Exit codes: 0 ok, 1 error, 2 usage, 3 ETL lease held elsewhere, 4 check failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from typing import Any, List, Optional


def _dump(obj: Any) -> str:
    """JSON for CLI output."""
    return json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))


def _settings():
    from .config import Settings

    return Settings.from_env()


def cmd_serve(args: argparse.Namespace) -> int:
    """Run uvicorn (single worker) with the background ETL loop."""
    import uvicorn

    from .api import create_app

    st = _settings()
    st.validate_for_serve()
    app = create_app(st, start_etl=not args.no_etl, pidfile=args.pidfile)
    pidfile = args.pidfile
    if pidfile:
        with open(pidfile, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
    try:
        uvicorn.run(app, host=st.bind_host, port=st.bind_port, workers=1, log_level="info",
                    server_header=False, proxy_headers=False, access_log=False)
    finally:
        if pidfile:
            try:
                os.remove(pidfile)
            except OSError:
                pass
    return 0


def cmd_etl(args: argparse.Namespace) -> int:
    """ETL entry points."""
    from .etl import Etl, LeaseHeld
    from .mongo import Mongo

    st = _settings()
    mongo = Mongo(st)
    etl = Etl(mongo, st)
    try:
        if args.full:
            print(_dump(etl.full()))
        elif args.reconcile:
            print(_dump(etl.reconcile()))
        elif args.loop:
            from . import indexes

            stop = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            indexes.ensure(mongo.qe)
            while not stop.is_set():
                try:
                    print(_dump(etl.run_once()), flush=True)
                except LeaseHeld as exc:
                    print(f"lease held: {exc}", file=sys.stderr)
                except Exception as exc:
                    print(f"ETL error: {type(exc).__name__}: {exc}", file=sys.stderr)
                stop.wait(st.etl_interval_s)
        else:
            print(_dump(etl.run_once()))
        if args.summary:
            print(_dump(etl.summary()))
        return 0
    except LeaseHeld as exc:
        print(f"lease held: {exc}", file=sys.stderr)
        return 3
    finally:
        mongo.close()


def cmd_indexes(args: argparse.Namespace) -> int:
    """indexes ensure|check|drop."""
    from . import indexes
    from .engine import QueryEngine
    from .mongo import Mongo

    st = _settings()
    mongo = Mongo(st)
    try:
        if args.action == "ensure":
            for line in indexes.ensure(mongo.qe):
                print(line)
            return 0
        if args.action == "drop":
            if not args.yes:
                print(f"refusing to drop indexes in {st.db} without --yes", file=sys.stderr)
                return 2
            for line in indexes.drop(mongo.qe):
                print(line)
            return 0
        eng = QueryEngine(st, mongo=mongo)
        ok, results = indexes.check(eng)
        for r in results:
            status = "PASS" if r["ok"] else "FAIL"
            print(f"{status} {r['shape']:<24} idx={','.join(r['indexes']) or '-':<32} "
                  f"docs={r['totalDocsExamined']} fetch={r['fetch_docsExamined']}/{r['fetch_nReturned']} "
                  f"nRet={r['nReturned']} ms={r['executionTimeMillis']} stages={'>'.join(s for s in r['stages'] if s)}"
                  + (f"  problems={r['problems']}" if r["problems"] else ""))
        print("indexes check:", "ALL PASS" if ok else "FAILED")
        return 0 if ok else 4
    finally:
        mongo.close()


def cmd_check(args: argparse.Namespace) -> int:
    """Configuration + connectivity self-check (no writes)."""
    from .aliases import get_registry
    from .mongo import Mongo

    problems: List[str] = []
    try:
        st = _settings()
    except Exception as exc:
        print(f"FAIL settings: {exc}")
        return 4
    print(f"ok   settings: source={st.source_db} qe_db={st.db} bind={st.bind} source_tz={st.source_tz} "
          f"display_tz={st.display_tz} filename_tz={st.filename_tz} reliable_from={st.stream_reliable_from} "
          f"token={'set' if st.api_token else 'EMPTY'} allow_no_token={st.allow_no_token}")
    try:
        st.validate_for_serve()
        print("ok   serve auth settings")
    except Exception as exc:
        problems.append(f"serve: {exc}")
        print(f"FAIL serve auth settings: {exc}")
    try:
        reg = get_registry()
        print(f"ok   aliases {reg.version}: {len(reg.metrics)} metrics")
    except Exception as exc:
        problems.append(f"aliases: {exc}")
        print(f"FAIL aliases: {exc}")
    mongo = Mongo(st)
    try:
        mongo.ping_source()
        n = mongo.source["screen_analysis"].estimated_document_count()
        print(f"ok   source reachable: screen_analysis ~{n} docs (read-only view)")
    except Exception as exc:
        problems.append(f"source: {exc}")
        print(f"FAIL source: {type(exc).__name__}: {exc}")
    try:
        mongo.ping_qe()
        meta = mongo.qe["meta"].find_one({"_id": "etl"}, max_time_ms=st.mongo_max_time_ms) or {}
        print(f"ok   qe db {st.db}: observations={meta.get('observations')} last_run_at={meta.get('last_run_at')} "
              f"last_ok={meta.get('last_ok')}")
    except Exception as exc:
        problems.append(f"qe: {exc}")
        print(f"FAIL qe db: {type(exc).__name__}: {exc}")
    finally:
        mongo.close()
    return 0 if not problems else 4


def build_parser() -> argparse.ArgumentParser:
    """argparse definition."""
    p = argparse.ArgumentParser(prog="python -m nicu_qe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the HTTP API (single worker) with the ETL loop")
    s.add_argument("--pidfile")
    s.add_argument("--no-etl", action="store_true", help="do not start the background ETL loop")
    s.set_defaults(fn=cmd_serve)
    e = sub.add_parser("etl", help="run the ETL")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="one incremental pass (default)")
    g.add_argument("--full", action="store_true", help="rebuild into observations_rebuild then rename")
    g.add_argument("--loop", action="store_true", help="incremental passes every QE_ETL_INTERVAL_S")
    g.add_argument("--reconcile", action="store_true", help="full in-place diff + orphan deletion")
    e.add_argument("--summary", action="store_true", help="print distribution counters afterwards")
    e.set_defaults(fn=cmd_etl)
    i = sub.add_parser("indexes", help="manage derived-DB indexes")
    i.add_argument("action", choices=["ensure", "check", "drop"])
    i.add_argument("--yes", action="store_true", help="confirm drop")
    i.set_defaults(fn=cmd_indexes)
    c = sub.add_parser("check", help="configuration and connectivity self-check")
    c.set_defaults(fn=cmd_check)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
