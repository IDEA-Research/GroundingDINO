# nicu_qe — MongoDB long-term metric query engine

The query engine reads `medical_monitor_db` on orin11 read-only. It writes normalised observations (canonical metric names and a true-UTC `observed_at`) into its own `medical_monitor_qe` DB and serves long-range queries from there. Specs: `../DESIGN.md` and `../API_CONTRACT.md` (in `hd-work/query-engine/`).

Status: WP-0 to WP-6 are done: pure-function layer (config, tz, alias tables, normalisation, time axis), ETL and indexes (WP-4), `QueryEngine` (WP-5), FastAPI + CLI (WP-6). Deployment files live in `deploy/` and `RUNBOOK.md` (WP-7).

## Layout

```
query_engine/
  pyproject.toml  requirements.txt  requirements-dev.txt  README.md
  nicu_qe/
    config.py       Settings.from_env(): QE_* environment variables with defaults
    tz.py           source wall-clock -> true UTC, display tz, ISO / step / duration parsing
    aliases.py      loads + validates data/*.json; normalize_key(), classify_device(), version
    normalize.py    normalize_medical_values() (values, composites, corrections), choose_primary()
    timeaxis.py     parse_video_name_time(), infer_source(), resolve_observed_at(), frame_offset_from_path()
    models.py       dataclasses: NormalizedValues, TimeResolution, Observation, Point, Series
    mongo.py        read-only source view (tz_aware=False, maxTimeMS, no write methods) + guarded QE DB handle
    etl.py          full rebuild / incremental / correction rescan / reconcile; daily_rollup, sessions, unmapped_keys
    indexes.py      declarative index list; ensure / drop / check (explain-based)
    engine.py       QueryEngine: catalog, coverage, series, latest, sessions, stats, unmapped_keys, aliases, health
    api.py          FastAPI shell (X-QE-Token, TrustedHost, JSON errors, asyncio.to_thread, in-process ETL loop)
    __main__.py     CLI
    data/metric_aliases.json  data/device_families.json
  tests/unit/         pure-function tests (no Mongo, no network)
  tests/integration/  live tests; run only with QE_TEST_MONGO=1 and QE_DB=medical_monitor_qe_test
```

## CLI

```
python -m nicu_qe check                      # settings + connectivity self-check (no writes)
python -m nicu_qe indexes ensure|check       # derived DB only; `indexes drop --yes` drops our named indexes
python -m nicu_qe etl --full                 # rebuild into observations_rebuild, then renameCollection(dropTarget)
python -m nicu_qe etl --once [--summary]     # one incremental pass (falls back to --full when no watermark)
python -m nicu_qe etl --loop | --reconcile
python -m nicu_qe serve [--pidfile F]        # uvicorn, single worker, background ETL loop holding the meta lease
```

`QE_DB` must match `^medical_monitor_qe(_test)?$` and differ from `QE_SOURCE_DB`, otherwise every command refuses to start. `serve` also needs `QE_API_TOKEN`, unless `QE_ALLOW_NO_TOKEN=1` and `QE_BIND` is a loopback address.

To add a new alias, edit `nicu_qe/data/metric_aliases.json`. The code does not hard-code any key name. Keys are normalised as `strip -> lower -> spaces/hyphens to "_"` and then looked up in the table for their device family. The loader rejects a malformed file, including an alias that maps to two canonicals within one family.

## Running the tests

Windows (Git Bash), from `hd-work/query-engine/`:

```
python -m venv .venv
.venv/Scripts/pip install -e "src/query_engine[dev]"
.venv/Scripts/python -m pytest src/query_engine/tests/unit -q
```

Linux (orin11), from the `query_engine/` directory:

```
/home/cluster/miniforge3/bin/python3.12 -m venv .venv-qe
.venv-qe/bin/pip install -r requirements-dev.txt -e .
.venv-qe/bin/python -m pytest tests/unit -q
```

Windows has no system IANA timezone database, so on Windows `tzdata` is installed automatically as a dependency. Linux uses the system database.

## Decisions not fixed by the spec

- `blood_pressure` with a bare scalar value (63 docs) goes to `unmapped`. It is not guessed as sys/dia/map.
- An explicit `bp_*` key wins over a part split out of the `blood_pressure` composite in the same doc.
- When two aliases of one canonical appear in the same doc, the first numeric value wins. If a later numeric value differs, the metric's `q` gets `alias_conflict` (a flag not in contract v1).
- Unknown family (contract v2 #14, aliases 2026-09-28.3, replaces the old `unknown_family_allowed` whitelist and WP-0's 1-character rule): an alias is accepted when its normalised text maps to one canonical across all families, is at least `unknown_family_min_len` (3) characters, is not in `unknown_family_deny` (default `rr, r, l, map, f, mv_set`) and is not a composite. So `bp_systolic`, `pip`, `FiO2`, `pmean`, `vt_meas`, `respiration_rate`, `heart_rate` map on unknown screens, while `MAP`/`map`/`RR`/`R`/`L`, the 2-character `HR`/`VT` and `blood_pressure` stay unmapped.
- `extra_family_aliases` on a metric registers aliases in another family's table: `hr` accepts `HR`/`heart_rate`/`hr` on osypka screens (2026-09-28.3).
- `fraction_to_percent` (≤1.0 → ×100) also applies to `fio2_set`.
- A correction to `null` removes the value from `m` but leaves `q[k] = ["corrected"]` so the removal stays auditable.
- `flags` includes `empty` only when medical_values was `{}`/`[]`/`None`/non-object and nothing survived the corrections.
- A stream doc with no `processed_at` uses `analyzed_at` (`capture`). A manual doc that has a file-name time but no frame offset falls back to `analysis`.

### WP-4 to WP-6 decisions

- Re-analysis (v2 M1): a stream doc is re-analysed when `analyzed_at > processed_at + 5 s`. Its time is restored from the earliest non-overwritten `analyzed_at` of a sibling screen of the same frame; that keeps `time_basis=capture` (flags `reanalyzed`, `time_from_sibling`). Without a sibling it becomes `capture_approx` from `processed_at`. The old symmetric 5-minute rule is gone, so a long LLM backlog no longer downgrades a correct capture time.
- `capture_legacy` only replaces `capture` (stream before `QE_STREAM_RELIABLE_FROM`); `capture_approx` stays as is.
- `future_time`: `observed_at > now + 1 day` falls back to `analyzed_at` with basis `analysis`; the flag goes to `flags` and every metric's `q`.
- `user_marked_bad` = source `has_glare` or `has_occlusion` is `true` (the only fields `update_quality_flags` writes besides `quality_marked_at`); it is added to every metric's `q` and to `flags`.
- Primary: `is_corrected > success > latest analyzed_at > larger _id`, recomputed for the whole (frame_result_id, original_screen_number ?? screen_number) group on every change. "Has values" is not a criterion (DESIGN 12.1 keeps the rule unchanged).
- `mk` lists every canonical with a numeric value regardless of quality; quality exclusion is a FETCH filter on `q.<metric>`, so `exclude_quality=` (empty) can still use the `mk` index.
- `daily_rollup` rows are per (display day, source, session, camera, device_family, time_basis, metric) over **primary** observations; the pseudo-metric `__all__` counts documents. Aggregates are recomputed for the touched days / sessions after each incremental run and fully (`$out`) after `--full` and reconcile.
- Coverage uses the rollup only when `group=day` and both `from` and `to` are display-tz midnights; otherwise it aggregates observations (exact half-open range).
- List endpoints (`stats`, `coverage`, `sessions`, `unmapped-keys`) return `{"rows": [...], "data_status": ...}` (plus metadata such as `time_axis`, `dedupe`, `note` for stats) instead of a bare array, so that `data_status` can always be attached (v2 #2).
- Percentiles use linear interpolation (numpy default). Above `QE_MAX_STATS_VALUES` stats fall back to Mongo `$group` for count/mean/min/max and return `p05/p50/p95 = null` with `note`.
- Series with `step`: buckets are `$dateTrunc` with `binSize` and the display tz; Python generates the same grid for `fill=null` (verified against Mongo in the integration tests). Point counts above `QE_MAX_POINTS` are truncated with `suggested_step`.
- `sessions` uses `{last_observed_at:1, first_observed_at:1}` for the interval-overlap query (contract v2 #15; `indexes ensure` drops the retired `{first_observed_at:1}`); results are sorted in Python (no blocking SORT).
- Incremental writes are diffed by a content hash `h` (etl_at excluded), so a no-op run writes nothing.
- Aliases 2026-09-28.2: `respiratory_rate` and `呼吸頻率` added to ventilator `vent_rr` (seen on ventilator screens). They are now ambiguous across families, so unknown devices no longer map `respiratory_rate` to `rr`.
