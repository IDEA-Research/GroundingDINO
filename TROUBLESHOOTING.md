# 🔧 醫療監控系統故障排除指南

**版本**: 1.0.0  
**最後更新**: 2025-11-18

本指南提供醫療監控系統常見問題的詳細解決方案和診斷步驟。

---

## 📋 快速診斷檢查清單

在詳細排查前，請先執行以下快速檢查：

```bash
# 1. 執行系統測試
./test_medical_monitoring.sh

# 2. 檢查服務狀態
curl -f http://localhost:8000/metrics > /dev/null && echo "✅ Medical Exporter OK" || echo "❌ Medical Exporter Failed"
curl -f http://localhost:9090/-/healthy > /dev/null && echo "✅ Prometheus OK" || echo "❌ Prometheus Failed"
curl -f http://localhost:3001/api/health > /dev/null && echo "✅ Grafana OK" || echo "❌ Grafana Failed"

# 3. 查看近期日誌
tail -20 medical_exporter.log
```

---

## 🗂️ 分類故障排除

### 1. MongoDB 相關問題

#### ❌ 問題: MongoDB 連接失敗
**錯誤訊息**:
```
MongoDB 連接失敗: ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused
```

**解決步驟**:

1. **檢查 MongoDB 服務狀態**
   ```bash
   # 方法1: 使用 mongo_manager 檢查
   python3 -c "from mongo_manager import get_mongo_manager; print('✅ MongoDB OK' if get_mongo_manager().is_mongodb_running() else '❌ MongoDB DOWN')"
   
   # 方法2: 直接檢查端口
   netstat -tuln | grep 27017 || echo "❌ MongoDB not listening on port 27017"
   ```

2. **啟動 MongoDB**
   ```bash
   # 使用內建管理器啟動
   python3 -c "from mongo_manager import start_local_mongodb; start_local_mongodb()"
   
   # 等待啟動完成
   sleep 5
   
   # 驗證啟動成功
   python3 -c "from mongo_manager import get_mongo_manager; get_mongo_manager().connect_to_mongodb()"
   ```

3. **檢查配置檔案**
   ```bash
   # 檢查 MongoDB 配置
   ls -la mongodb_config/mongod.conf
   cat mongodb_config/mongod.conf
   
   # 檢查數據目錄權限
   ls -la mongodb_data/
   ```

4. **手動啟動 MongoDB** (如果自動啟動失敗)
   ```bash
   # 確保目錄存在
   mkdir -p mongodb_data/db mongodb_data/logs
   
   # 手動啟動
   mongod --config mongodb_config/mongod.conf
   ```

#### ❌ 問題: MongoDB 數據查詢緩慢
**症狀**: 查詢時間 > 5秒

**解決步驟**:

1. **檢查索引**
   ```javascript
   // 連接 MongoDB
   mongo mongodb://localhost:27017/medical_monitor_db
   
   // 檢查索引
   db.screen_analysis.getIndexes()
   db.frame_results.getIndexes()
   db.video_analysis.getIndexes()
   ```

2. **重建索引**
   ```bash
   python3 -c "
   from mongo_manager import get_mongo_manager
   mgr = get_mongo_manager()
   mgr.connect_to_mongodb()
   mgr.create_indexes()
   print('✅ 索引重建完成')
   "
   ```

3. **檢查數據庫大小**
   ```javascript
   db.stats()
   db.screen_analysis.count()
   ```

---

### 2. Medical Exporter 相關問題

#### ❌ 問題: Medical Exporter 無法啟動
**錯誤訊息**:
```
OSError: [Errno 98] Address already in use
```

**解決步驟**:

1. **檢查端口佔用**
   ```bash
   # 查看佔用 port 8000 的進程
   netstat -tuln | grep 8000
   lsof -i :8000
   
   # 查找 Medical Exporter 進程
   ps aux | grep medical_mongodb_exporter
   ```

2. **終止現有進程**
   ```bash
   # 優雅終止
   if [ -f medical_exporter.pid ]; then
       kill $(cat medical_exporter.pid)
       rm medical_exporter.pid
   fi
   
   # 強制終止（如果需要）
   pkill -f medical_mongodb_exporter.py
   ```

3. **重新啟動**
   ```bash
   # 手動啟動以查看詳細錯誤
   python3 medical_mongodb_exporter.py --log-level DEBUG
   ```

#### ❌ 問題: 指標數據不更新
**症狀**: Prometheus 指標長時間無變化

**診斷步驟**:

1. **檢查 Exporter 日誌**
   ```bash
   tail -f medical_exporter.log
   
   # 查看錯誤日誌
   grep "ERROR" medical_exporter.log
   grep "WARNING" medical_exporter.log
   ```

2. **驗證 MongoDB 數據**
   ```bash
   python3 -c "
   from mongo_manager import get_mongo_manager
   mgr = get_mongo_manager()
   mgr.connect_to_mongodb()
   
   # 檢查最新數據
   from datetime import datetime, timedelta
   cutoff = datetime.now() - timedelta(minutes=10)
   count = mgr.db.screen_analysis.count_documents({
       'analyzed_at': {'\$gte': cutoff},
       'success': True,
       'medical_values': {'\$ne': {}}
   })
   print(f'最近10分鐘醫療數據: {count} 筆')
   "
   ```

3. **重啟 Exporter**
   ```bash
   ./stop_medical_monitoring.sh
   sleep 5
   ./start_medical_monitoring.sh
   ```

#### ❌ 問題: 異常檢測不工作
**症狀**: 明顯異常數值沒有觸發警報

**檢查步驟**:

1. **驗證異常檢測邏輯**
   ```bash
   python3 -c "
   from medical_mongodb_exporter import MedicalMongoExporter
   exporter = MedicalMongoExporter()
   
   # 測試異常檢測
   test_values = {'heart_rate': 35, 'spo2': 85}  # 異常數值
   exporter._detect_and_record_abnormalities(test_values, 'test_session', 'test_device')
   print('✅ 異常檢測測試完成')
   "
   ```

2. **檢查警報計數器**
   ```bash
   curl -s http://localhost:8000/metrics | grep medical_alerts_total
   ```

---

### 3. Prometheus 相關問題

#### ❌ 問題: Prometheus 無法抓取指標
**錯誤訊息**: Target 顯示 "Connection refused"

**解決步驟**:

1. **檢查目標狀態**
   ```bash
   # 訪問 Prometheus 目標頁面
   curl http://localhost:9090/targets
   
   # 或直接訪問 Web UI
   python3 -c "import webbrowser; webbrowser.open('http://localhost:9090/targets')"
   ```

2. **驗證網路連通性**
   ```bash
   # 從 Prometheus 容器測試連通性
   docker exec medical-prometheus wget -qO- http://host.docker.internal:8000/metrics
   
   # 檢查防火牆設定
   sudo ufw status
   ```

3. **檢查 Prometheus 配置**
   ```bash
   # 驗證配置檔案語法
   docker exec medical-prometheus promtool check config /etc/prometheus/prometheus.yml
   
   # 重載配置
   curl -X POST http://localhost:9090/-/reload
   ```

#### ❌ 問題: 警報規則不生效
**症狀**: 異常數值沒有觸發 Prometheus 警報

**檢查步驟**:

1. **驗證警報規則語法**
   ```bash
   # 檢查警報規則檔案
   docker exec medical-prometheus promtool check rules /etc/prometheus/alerts/medical_alerts.yml
   ```

2. **檢查警報狀態**
   ```bash
   # 訪問警報頁面
   curl http://localhost:9090/alerts
   
   # 或使用 Web UI
   python3 -c "import webbrowser; webbrowser.open('http://localhost:9090/alerts')"
   ```

3. **測試警報查詢**
   ```bash
   # 手動測試警報條件
   curl -G 'http://localhost:9090/api/v1/query' \
     --data-urlencode 'query=patient_heart_rate_bpm < 40 or patient_heart_rate_bpm > 180'
   ```

---

### 4. Grafana 相關問題

#### ❌ 問題: Grafana 無法訪問
**錯誤訊息**: "This site can't be reached"

**解決步驟**:

1. **檢查容器狀態**
   ```bash
   docker-compose -f docker-compose.monitoring.yml ps
   docker logs medical-grafana
   ```

2. **檢查端口映射**
   ```bash
   docker port medical-grafana
   netstat -tuln | grep 3001
   ```

3. **重啟 Grafana**
   ```bash
   docker-compose -f docker-compose.monitoring.yml restart grafana
   
   # 等待啟動
   sleep 10
   curl -f http://localhost:3001/api/health
   ```

#### ❌ 問題: 儀表板顯示 "No data"
**症狀**: Grafana 面板沒有數據

**診斷步驟**:

1. **檢查數據源連接**
   ```bash
   # 使用 Grafana API 檢查數據源
   curl -u admin:medical123 http://localhost:3001/api/datasources
   
   # 測試數據源連通性
   curl -u admin:medical123 -X POST http://localhost:3001/api/datasources/proxy/1/api/v1/query\?query=up
   ```

2. **驗證查詢語法**
   - 在 Grafana Explore 頁面測試 PromQL 查詢
   - 檢查時間範圍設定
   - 驗證標籤篩選器正確性

3. **檢查數據時間範圍**
   ```bash
   # 查看 Prometheus 中的數據時間範圍
   curl -G 'http://localhost:9090/api/v1/query' \
     --data-urlencode 'query=patient_heart_rate_bpm' | jq '.data.result[].value[0]'
   ```

---

### 5. Docker 相關問題

#### ❌ 問題: Docker 容器啟動失敗
**錯誤訊息**: "Container exited with code 1"

**解決步驟**:

1. **檢查容器日誌**
   ```bash
   docker-compose -f docker-compose.monitoring.yml logs --tail=50
   
   # 查看特定服務日誌
   docker-compose -f docker-compose.monitoring.yml logs prometheus
   docker-compose -f docker-compose.monitoring.yml logs grafana
   ```

2. **檢查資源使用**
   ```bash
   # 檢查磁盤空間
   df -h
   
   # 檢查記憶體使用
   free -h
   
   # 檢查 Docker 資源
   docker system df
   ```

3. **清理並重建**
   ```bash
   # 停止並清理
   docker-compose -f docker-compose.monitoring.yml down -v
   
   # 清理 Docker 資源
   docker system prune -f
   
   # 重新啟動
   ./start_medical_monitoring.sh
   ```

---

## 🔍 進階診斷工具

### 系統健康檢查腳本

創建一個全面的健康檢查腳本：

```bash
#!/bin/bash
# health_check.sh

echo "🔍 醫療監控系統健康檢查"
echo "=========================="

# 1. 檢查基礎服務
echo "1. 基礎服務檢查:"
python3 -c "from mongo_manager import get_mongo_manager; print('✅ MongoDB' if get_mongo_manager().is_mongodb_running() else '❌ MongoDB')"

curl -s -f http://localhost:8000/metrics > /dev/null && echo "✅ Medical Exporter" || echo "❌ Medical Exporter"
curl -s -f http://localhost:9090/-/healthy > /dev/null && echo "✅ Prometheus" || echo "❌ Prometheus"
curl -s -f http://localhost:3001/api/health > /dev/null && echo "✅ Grafana" || echo "❌ Grafana"

# 2. 數據流檢查
echo -e "\n2. 數據流檢查:"
python3 -c "
from mongo_manager import get_mongo_manager
from datetime import datetime, timedelta
mgr = get_mongo_manager()
mgr.connect_to_mongodb()
cutoff = datetime.now() - timedelta(hours=1)
count = mgr.db.screen_analysis.count_documents({'analyzed_at': {'\$gte': cutoff}})
print(f'最近1小時醫療數據: {count} 筆')
"

# 3. 指標檢查
echo -e "\n3. Prometheus 指標檢查:"
metrics_count=$(curl -s http://localhost:8000/metrics | grep -c "^patient_")
echo "患者指標數量: $metrics_count"

# 4. 警報檢查
echo -e "\n4. 警報狀態檢查:"
alerts_count=$(curl -s 'http://localhost:9090/api/v1/alerts' | jq '.data.alerts | length' 2>/dev/null || echo "0")
echo "活躍警報數量: $alerts_count"

echo -e "\n=========================="
echo "健康檢查完成"
```

### 效能監控腳本

```bash
#!/bin/bash
# performance_check.sh

echo "📊 系統效能監控"
echo "================"

# MongoDB 查詢效能
echo "1. MongoDB 查詢效能:"
python3 -c "
import time
from mongo_manager import get_mongo_manager
mgr = get_mongo_manager()
mgr.connect_to_mongodb()

start = time.time()
list(mgr.db.screen_analysis.find().limit(100))
end = time.time()
print(f'查詢100筆記錄耗時: {(end-start)*1000:.2f}ms')
"

# Medical Exporter 響應時間
echo -e "\n2. Medical Exporter 響應時間:"
response_time=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/metrics)
echo "Metrics 端點響應時間: ${response_time}s"

# 記憶體使用
echo -e "\n3. 記憶體使用:"
exporter_pid=$(pgrep -f medical_mongodb_exporter.py)
if [ -n "$exporter_pid" ]; then
    memory_mb=$(ps -p $exporter_pid -o rss= | awk '{print $1/1024}')
    echo "Medical Exporter 記憶體: ${memory_mb}MB"
fi

# Docker 容器資源使用
echo -e "\n4. Docker 容器資源:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" medical-prometheus medical-grafana 2>/dev/null || echo "Docker 容器未運行"
```

---

## 📞 支援資源

### 日誌位置

```bash
# 系統日誌
./medical_exporter.log          # Medical Exporter 主日誌
./test_video_processing.log     # 影片處理測試日誌

# Docker 日誌
docker-compose -f docker-compose.monitoring.yml logs prometheus
docker-compose -f docker-compose.monitoring.yml logs grafana

# MongoDB 日誌
./mongodb_data/logs/mongod.log  # MongoDB 服務日誌
```

### 重要配置檔案

```bash
./prometheus.yml                # Prometheus 主配置
./alerts/medical_alerts.yml     # 警報規則
./docker-compose.monitoring.yml # Docker 編排
./grafana/provisioning/         # Grafana 配置
```

### 重啟服務的正確順序

```bash
# 完全重啟系統
./stop_medical_monitoring.sh
sleep 10
./start_medical_monitoring.sh

# 或分步重啟
# 1. 停止監控服務
docker-compose -f docker-compose.monitoring.yml down

# 2. 重啟 Medical Exporter
pkill -f medical_mongodb_exporter.py
sleep 5
python3 medical_mongodb_exporter.py &

# 3. 重啟監控服務
docker-compose -f docker-compose.monitoring.yml up -d
```

### 緊急恢復程序

如果系統完全無響應：

```bash
# 1. 強制停止所有服務
./stop_medical_monitoring.sh
docker-compose -f docker-compose.monitoring.yml down -v
pkill -9 -f medical_mongodb_exporter.py
pkill -9 -f mongod

# 2. 清理臨時檔案
rm -f medical_exporter.pid medical_exporter.log
rm -rf prometheus-data grafana-data

# 3. 重新啟動
./start_medical_monitoring.sh

# 4. 驗證恢復
./test_medical_monitoring.sh
```

---

## 📋 常見錯誤代碼

| 錯誤代碼 | 描述 | 解決方案 |
|---------|------|----------|
| `111` | Connection refused | 檢查服務是否運行，端口是否正確 |
| `98` | Address already in use | 終止佔用端口的進程 |
| `13` | Permission denied | 檢查檔案權限，可能需要 sudo |
| `2` | No such file or directory | 檢查檔案路徑是否正確 |
| `timeout` | 連接超時 | 檢查網路連通性，增加超時時間 |

---

**提醒**: 如果以上方法都無法解決問題，請執行 `./test_medical_monitoring.sh` 獲得完整的系統診斷報告，或聯繫技術支援團隊。

---

**最後更新**: 2025-11-18  
**文檔版本**: 1.0.0