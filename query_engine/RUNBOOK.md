# RUNBOOK — medical-query-engine（nicu_qe）on orin11

適用：orin11（Jetson、Ubuntu 22.04、systemd 249）。依據：`DESIGN.md` §8／§12.5、`API_CONTRACT.md` §0／§6、`REVIEW_DESIGN_2026-09-28.md` M6–M10、L7、L8、L11。
**orin11 的 sudo 需要密碼：所有 `sudo` 指令都由使用者本人執行。** agent 只負責 sync 與唯讀查證。

---

## 1. 概述與架構

nicu_qe 從 live 醫療 DB `medical_monitor_db` **唯讀**抽取 screen analysis，正規化指標名稱與時間軸後寫入自有 DB `medical_monitor_qe`，並在 `127.0.0.1:8020` 提供唯讀 HTTP API。

```
 medical-ai-analysis / web_app ──寫入──▶ ┌────────────────────────────────────┐
                                         │ docker: medical-mongodb            │
                                         │ host network 127.0.0.1:27017 無 auth│
                                         │  ├─ medical_monitor_db  (來源，唯讀) │
                                         │  └─ medical_monitor_qe  (本服務自有) │
                                         └───────▲──────────────┬─────────────┘
                              find/aggregate（唯讀）│              │ ETL 寫入 / 查詢
                                         ┌───────┴──────────────▼─────────────┐
                                         │ medical-query-engine.service        │
                                         │ User=cluster                        │
                                         │ /home/cluster/services/query_engine │
                                         │ .venv-qe/bin/python -m nicu_qe serve│
                                         │  ├─ 背景 ETL（watermark 增量）       │
                                         │  └─ FastAPI 127.0.0.1:8020          │
                                         └───────▲────────────────────────────┘
                     X-QE-Token（/v1/*）          │ HTTP loopback
                         helper-dashboard（:8010 backend）、db-agent
```

| 項目 | 值 |
|---|---|
| unit | `/etc/systemd/system/medical-query-engine.service`（`deploy/medical-query-engine.service`） |
| 程式目錄 | `/home/cluster/services/query_engine/`（**git 外**，避免 `git clean`／切分支，M10） |
| venv | `/home/cluster/services/query_engine/.venv-qe`（base：`/home/cluster/miniforge3/bin/python3.12`） |
| 環境檔 | `/home/cluster/services/query_engine/.env.qe`（600，owner cluster；範本 `deploy/env.qe.example`） |
| staging | `/home/cluster/qe-stage/query_engine/`（sync 目的地；install 從這裡 rsync 到程式目錄） |
| sudoers | `/etc/sudoers.d/cluster-medical-query-engine`（cluster 免密 start/stop/restart/status/reset-failed 本 unit） |
| port | 8020（loopback）；臨時驗證只用 8021 |
| 保護 | `StartLimitIntervalSec=600`／`StartLimitBurst=5`（在 `[Unit]`）、無 `ExecStartPre`、`ProtectSystem=strict` |

## 2. 前置作業（部署前一次做完）

1. **備份來源 DB**（約 32 MB，秒級）。2026-09-28 實查：orin11 **host 上沒有** `mongodump`／`mongosh`（`which` 無結果），要透過容器執行，直接串流到 host 檔案：
   ```bash
   mkdir -p ~/backup-$(date +%F)
   docker exec medical-mongodb mongodump --db medical_monitor_db --archive --gzip \
     > ~/backup-$(date +%F)/medical_monitor_db.archive.gz
   ls -lh ~/backup-$(date +%F)/
   ```
   若日後 host 裝了 database tools，等價指令為 `mongodump --db medical_monitor_db --out ~/backup-$(date +%F)`。
   還原（只在災難時）：`docker exec -i medical-mongodb mongorestore --archive --gzip < ~/backup-…/medical_monitor_db.archive.gz`。
2. 確認 Mongo 容器在跑：`docker ps --filter name=medical-mongodb`（STATUS 應為 `Up … (healthy)`）。
3. 確認 8020 空：`ss -ltn | grep -E ':(8020|8021) '` 應無輸出。
4. 使用者已同意（DESIGN §12.6）：正式 ETL 寫 `medical_monitor_qe`、親自執行 `sudo install.sh`。

## 3. 部署步驟

1. **Sync（Windows Git Bash，agent 或使用者執行）**
   ```bash
   bash /c/Users/flash/hd-work/query-engine/src/query_engine/deploy/sync-from-windows.sh
   ```
   以 `tar | ssh` 把 `src/query_engine/` 送到 orin11 `~/qe-stage/query_engine/`（排除 `.venv*`、`*.egg-info`、`.pytest_cache`、`__pycache__`、`.env.qe`、`*.pid`）。不碰 git root、不碰 services 目錄。
2. **使用者先讀腳本**（L8：腳本放在 cluster 可寫的目錄，sudo 前必須看過）：
   ```bash
   less ~/qe-stage/query_engine/deploy/install.sh
   less ~/qe-stage/query_engine/deploy/medical-query-engine.service
   less ~/qe-stage/query_engine/deploy/sudoers-cluster-medical-query-engine
   ```
3. **使用者執行安裝**：
   ```bash
   sudo bash ~/qe-stage/query_engine/deploy/install.sh ~/qe-stage/query_engine
   ```
   腳本步驟（每步都會印 `==> [n]`，任何一步失敗即停）：0 前置檢查（root、`SUDO_USER=cluster`）→ 1 rsync 到程式目錄（保留 `.venv-qe`、`.env.qe`）→ 2 建 venv＋`pip install -e`（失敗則以 `--system-site-packages` 重建並警告）→ 3 沒有 `.env.qe` 才產生（含新 token）；檢查 `QE_DB` 合法 → 4 安裝 unit → 5 `visudo -cf` 後安裝 sudoers → 6 `daemon-reload` → 7 port 檢查（被別人佔就中止，**不殺任何進程**）→ 8 `enable`＋`start`（已在跑則 `restart`）→ 9 等 10 秒打 `/health` → 10 印 `NRestarts` 與下一步。
   **可重跑**：升級程式時重做步驟 1 與 3 即可；`.env.qe` 與 token 不會被覆蓋。要強制重裝套件：`sudo QE_FORCE_PIP=1 bash …/install.sh …`。
4. **驗證**（cluster 身分，不需 sudo）：
   ```bash
   curl -s http://127.0.0.1:8020/health; echo
   cd /home/cluster/services/query_engine
   ( set -a; . ./.env.qe; set +a; .venv-qe/bin/python -m nicu_qe indexes check ); echo "exit=$?"
   systemctl show -p NRestarts -p ActiveState medical-query-engine
   ```
   - `/health` 應 HTTP 200、`status:"ok"`、`db:"ok"`、`qe_db:"ok"`。
   - `indexes check` exit 0（任何 COLLSCAN 或 `totalDocsExamined > 3×nReturned` 會非零退出）。
   - ETL 狀態：首次啟動會從空 watermark 追到最新；幾分鐘後 `etl.last_ok:true`、`etl.observations` 接近原庫 screen analysis 筆數、`etl.lag_seconds` 小於數個 `QE_ETL_INTERVAL_S`。
   - `NRestarts=0`。
   - 帶 token 試一個 `/v1/*`：`curl -s -H "X-QE-Token: $(grep '^QE_API_TOKEN=' .env.qe | cut -d= -f2-)" http://127.0.0.1:8020/v1/catalog | head -c 400`。

## 4. 把 token 交給消費者

token 只存在 `/home/cluster/services/query_engine/.env.qe`（600）。讀取：`grep '^QE_API_TOKEN=' /home/cluster/services/query_engine/.env.qe`。
在消費者（helper-dashboard backend、db-agent）的 env 檔加入：
```
HELPER_DASHBOARD_QUERY_ENGINE_URL=http://127.0.0.1:8020
HELPER_DASHBOARD_QUERY_ENGINE_TOKEN=<上面的值>
```
helper-dashboard 的 env 檔是 `helper-dashboard/.env`（unit `medical-helper-dashboard` 的 `EnvironmentFile`）。改完需重啟該消費者；**注意 orin11 dashboard 的 port 衝突禁令**：只用 `sudo systemctl restart medical-helper-dashboard`，不要手動跑 `dev.sh`／`next -p 3050`。
換 token：編輯 `.env.qe` 的 `QE_API_TOKEN`（`openssl rand -hex 32`）→ `sudo systemctl restart medical-query-engine` → 同步更新所有消費者。token 不要貼進聊天、commit 或 log。

## 5. 日常操作

| 要做的事 | 指令（cluster 身分） |
|---|---|
| 狀態 | `systemctl status medical-query-engine`（cluster 可直接看；或 `sudo systemctl status …` 走免密規則） |
| 即時 log | `journalctl -u medical-query-engine -f`（cluster 在 `adm` 群組，免 sudo） |
| 近一小時 log | `journalctl -u medical-query-engine --since -1h --no-pager` |
| 重啟 | `sudo systemctl restart medical-query-engine`（免密） |
| 重啟計數 | `systemctl show -p NRestarts medical-query-engine`（持續增加＝有問題） |
| 健康 | `curl -s http://127.0.0.1:8020/health` |
| 手動跑 CLI | `cd /home/cluster/services/query_engine && ( set -a; . ./.env.qe; set +a; .venv-qe/bin/python -m nicu_qe <子命令> )` |

CLI 子命令（契約）：`serve`、`etl --once|--full|--loop`、`indexes ensure|check|drop`、`check`。

**`/health` 欄位怎麼讀**

| 欄位 | 意義 | 正常值 |
|---|---|---|
| `status` | `ok`／`degraded`（ETL 失敗，HTTP 仍 200）／`error`（DB 連不上，HTTP 503） | `ok` |
| `db`、`qe_db` | 來源 DB／自有 DB 連線 | `ok` |
| `etl.last_run_at` | 上次 ETL 執行時間（+08:00） | 距今 < 數分鐘 |
| `etl.last_ok`、`etl.last_error` | 上次 ETL 是否成功與錯誤訊息 | `true`、`null` |
| `etl.watermark_id` | 已處理到的 `screen_analysis._id` | 隨時間前進 |
| `etl.lag_seconds` | 原庫最新資料與 watermark 的落差 | 小於數個 `QE_ETL_INTERVAL_S`（60 s） |
| `etl.observations`、`etl.sessions` | 自有 DB 筆數 | 單調增加 |
| `source_tz`、`source_tz_offset_hours_detected` | 來源時區與實測 offset | `Asia/Taipei`、≈ `8.0` |
| `aliases_version`、`version` | 別名表版本、程式版本 | 與部署版本一致 |

**lag 過大怎麼辦**：先看 `etl.last_error` 與 `journalctl`。(1) 若 `last_ok:false` → 依錯誤處理（常見：Mongo 暫時不可用，恢復後自動追上）。(2) 若 `last_ok:true` 但 lag 持續增加 → 可能 ETL 被 lease 卡住（見 §7）或寫入量暴增；可手動 `etl --once` 觀察耗時。(3) 原庫長時間沒有新資料時，lag 以原庫最新資料為準，不代表異常。(4) 必要時 `etl --full` 全量重建（約 28k 筆，預估 < 1 分鐘）。

## 6. 新增別名流程

1. 在 Windows 的 `src/query_engine/nicu_qe/data/metric_aliases.json` 加入別名，並把 `version` 改成新值（例 `2026-10-05.1`）；可用 `/v1/unmapped-keys` 找出待對應的原始鍵名。
2. 本機跑單元測試（`pytest tests/unit`）。
3. `sync-from-windows.sh` → 使用者 `sudo bash ~/qe-stage/query_engine/deploy/install.sh ~/qe-stage/query_engine`（會 rsync 新 JSON 並 restart）。
4. 讓既有資料套用新別名：
   ```bash
   cd /home/cluster/services/query_engine
   ( set -a; . ./.env.qe; set +a; .venv-qe/bin/python -m nicu_qe etl --full )
   ```
5. 確認 `/health` 的 `aliases_version` 已更新、`/v1/unmapped-keys` 中該鍵消失。

## 7. 回滾

| 情境 | 做法 | 資料 |
|---|---|---|
| 暫停服務 | `bash /home/cluster/services/query_engine/deploy/rollback.sh --yes` | 全保留 |
| 停服務並丟掉衍生資料 | `bash …/rollback.sh --drop-db`（先看計畫）→ `bash …/rollback.sh --drop-db --yes` | 只 drop `medical_monitor_qe` |
| 丟測試庫 | `bash …/rollback.sh --drop-db --db medical_monitor_qe_test --yes` | 只 drop `_test` |
| 完全移除 unit | `sudo bash …/deploy/uninstall.sh` | 程式目錄、`.env.qe`、DB 全保留 |

- `rollback.sh` 不加 `--yes` 只印計畫（dry run）。DB 名在 bash（`^medical_monitor_qe(_test)?$`）與 mongosh 內 JS regex **兩層**斷言，`medical_monitor_db` 無法被 drop。**不要手打 mongosh 的 `dropDatabase()`**（M6）。
- `uninstall.sh`：stop → disable → 移除 unit 與 sudoers → `daemon-reload` → `reset-failed`；最後印出手動刪除指令（`rm -rf -- /home/cluster/services/query_engine`、`rollback.sh --drop-db --yes`）。若要 drop DB，**先**跑 rollback.sh 再刪程式目錄。
- 來源 DB `medical_monitor_db` 在任何回滾路徑都不受影響；其他既有服務零影響。

## 8. 故障排除

- **port 8020 被佔**：`install.sh` 會中止並列出佔用者，不會殺任何進程。查 `sudo ss -ltnp 'sport = :8020'`，確認是誰（常見是忘了關的臨時驗證進程；臨時驗證規定只能用 8021）。由使用者決定是否停掉該進程，或改 `.env.qe` 的 `QE_BIND` 後重跑 install。
- **StartLimit 觸發**（600 秒內啟動 5 次失敗 → `Active: failed (Result: start-limit-hit)`）：這是刻意的煞車，避免重演 2026-09-10 的 crash loop。先 `journalctl -u medical-query-engine -n 100 --no-pager` 找根因並修好，再 `sudo systemctl reset-failed medical-query-engine && sudo systemctl start medical-query-engine`（兩者都在免密規則內）。
- **Mongo 容器沒起來時 `/health` 回 503**：屬正常設計。程式不會因 Mongo 不可用而退出（不會燒 StartLimit），會在背景重試連線與 `indexes ensure`。處理 Mongo：`docker ps -a --filter name=medical-mongodb`、`docker logs --tail 50 medical-mongodb`；容器恢復後 `/health` 自動轉 200。
- **ETL lease 卡住**（`etl.last_run_at` 不再前進、log 顯示 lease 被占用）：常見原因是另一個 ETL（手動 `etl --loop`／`--full`）還在跑，或前一個進程被強殺留下未過期的 lease。先確認沒有手動 ETL：`pgrep -af 'nicu_qe etl'`；有就讓它跑完或由使用者停掉。沒有的話等 lease 過期後自動恢復；仍卡住則 `sudo systemctl restart medical-query-engine`。不要手動改 `medical_monitor_qe` 內的 lease／meta 文件。
- **時區看起來差 8 小時**：原庫所有 Date 是「台北牆鐘時間被當成 UTC 存」，所以直接用 mongosh 看原庫會看到 `…Z` 但其實是台北時間。本服務以 `QE_SOURCE_TZ` 還原成真 UTC：HTTP 回應帶 `+08:00`（display tz），Python library 回 UTC aware datetime（兩者同一時刻）。判讀步驟：(1) `/health` 的 `source_tz_offset_hours_detected` 應 ≈ 8.0，偏離代表原庫寫入端時區變了，需回報；(2) 比對時確認看的是 `+08:00` 還是 `Z`；(3) 不帶時區的查詢參數一律視為台北時間。
- **`/health` 正常但 `/v1/*` 被拒（4xx）**：消費者沒帶或帶錯 `X-QE-Token`；Host header 不是 `127.0.0.1`／`localhost` 也會被 TrustedHost 擋。
- **服務啟動即退出、log 顯示 QE_DB 相關錯誤**：`QE_DB` 不符 `^medical_monitor_qe(_test)?$` 或等於 `QE_SOURCE_DB`，屬啟動保護，修 `.env.qe` 後重啟。
- **pip 失敗後用了 `--system-site-packages`**：install 會印 WARNING；服務改用 miniforge base 的 fastapi／uvicorn／pydantic／pymongo。base 環境被升級時可能連帶影響本服務；網路恢復後可 `sudo QE_FORCE_PIP=1 bash …/install.sh …` 重裝。

## 9. 安全注意

- **token**：`QE_API_TOKEN` 預設必填，`/v1/*` 需 `X-QE-Token`。`.env.qe` 權限 600；不要把 token 放進 git、聊天或 log。orin11 是多使用者機器（例如 :8090 屬其他使用者），loopback 不等於安全，所以 token 不可省略；`QE_ALLOW_NO_TOKEN=1` 僅限本機測試。
- **loopback**：`QE_BIND` 保持 `127.0.0.1:8020`。不要綁 `0.0.0.0` 或區網 IP。
- **絕不把 `QE_DB` 設成 `medical_monitor_db`**。程式啟動、install.sh、rollback.sh 都有 DB 名斷言，但這是最後一道防線，不是可以放心試錯的理由。Mongo 沒有 auth。
- 服務對來源 DB 只做 find／aggregate（read-only wrapper）；不在原庫建索引、不改文件。
- sudoers 片段只放行本 unit 的固定 systemctl 參數（start/stop/restart/status/reset-failed），完整路徑 `/usr/bin/systemctl`（2026-09-28 在 orin11 實查）；不放行 journalctl——cluster 在 `adm` 群組，直接 `journalctl -u medical-query-engine -f` 即可。
- unit 硬化：`NoNewPrivileges`、`PrivateTmp`、`ProtectSystem=strict`（只有程式目錄可寫）。
- `install.sh` 放在 cluster 可寫的 staging 目錄，任何人能改它；每次 `sudo` 前都要重新讀過。

## 10. 已知限制

- **dataset 來源未實測**：`source=dataset` 的時間軸（檔名解析）與正規化只以程式邏輯與少量樣本推定，尚未在真實 dataset 上傳資料驗證。
- **檔名時間假設**：manual／dataset 的 `observed_at` = 檔名日期時間（以 `QE_FILENAME_TZ` 解讀）＋ frame 偏移秒數。若影片檔名不是開始錄影時間，或影片經過剪輯（開頭被剪掉），時間會整段偏移；檔名含 `QE_EDITED_MARKERS` 者標 `time_confidence=low` 與 `edited_video`，其餘剪輯無法自動偵測。
- **reanalyze 偵測誤差**：stream 資料的 `analyzed_at` 若被重新分析覆寫，改用 `frame_results.processed_at`（`capture_approx`，誤差約 +20 秒）；判定閾值為 `analyzed_at > processed_at + 5 s`，邊界附近可能誤判。
- `QE_STREAM_RELIABLE_FROM`（2026-06-01）之前的 stream 資料標 `capture_legacy`（confidence medium）。
- pyproject 目前依賴 `uvicorn[standard]`（L11 建議改 `uvicorn`）；在 aarch64 會多裝 uvloop／httptools／watchfiles，裝不起來時走 `--system-site-packages` fallback。
