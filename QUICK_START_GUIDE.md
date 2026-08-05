# 🏥 醫療監控系統 - 快速使用指南

## 🚀 **30秒快速啟動**

### 第一步：啟動監控系統
```bash
# 一鍵啟動所有服務
./start_medical_monitoring.sh
```

### 第二步：處理醫療影片（可選）
```bash
# 處理測試影片產生醫療數據
python3 video_screen_digit_extractor.py \
  --video_path medSample/med20250812-9.mkv \
  --api_key "你的OpenAI_API_KEY" \
  --target_data medical_values
```

### 第三步：查看監控結果
**打開瀏覽器訪問：**
- 📊 **Grafana 儀表板**：http://localhost:3001
  - 帳號：`admin`
  - 密碼：`medical123`
- 📈 **Prometheus 數據**：http://localhost:9090  
- 🔧 **原始指標**：http://localhost:8000/metrics

---

## 📋 **詳細使用步驟**

### **步驟1：環境檢查**
```bash
# 檢查系統是否就緒
python3 --version                    # 確認 Python 3.8+
ls medical_mongodb_exporter.py       # 確認核心檔案存在
```

### **步驟2：啟動服務**

**方法A：使用自動化腳本（推薦）**
```bash
# 啟動完整監控系統
./start_medical_monitoring.sh

# 如果需要停止服務
./stop_medical_monitoring.sh
```

**方法B：手動啟動（僅監控服務）**
```bash
# 僅啟動 MongoDB 和 Medical Exporter
python3 medical_mongodb_exporter.py --port 8000 --interval 15
```

### **步驟3：驗證系統運行**
```bash
# 檢查服務狀態  
curl -s http://localhost:8000/metrics | grep patient_heart_rate_bpm
```

### **步驟4：處理醫療視頻**

**使用現有測試影片：**
```bash
python3 video_screen_digit_extractor.py \
  --video_path medSample/med20250812-9.mkv \
  --api_key "sk-proj-你的完整OpenAI_API_KEY" \
  --output_dir monitoring_data \
  --target_data medical_values
```

**處理你的醫療影片：**
```bash
python3 video_screen_digit_extractor.py \
  --video_path "路徑/到/你的/醫療影片.mp4" \
  --api_key "你的OpenAI_API_KEY" \
  --target_data medical_values
```

### **步驟5：查看監控結果**

**1. Grafana 儀表板（最直觀）**
- 訪問：http://localhost:3001  
- 登入：`admin` / `medical123`
- 查看：
  - 患者監控儀表板 (Patient Monitoring)
  - 系統總覽儀表板 (System Overview)

**2. Prometheus 查詢**
- 訪問：http://localhost:9090
- 查詢範例：
  ```
  patient_heart_rate_bpm                # 患者心率
  patient_spo2_percentage               # 血氧飽和度  
  medical_alerts_total                  # 醫療警報總數
  ```

**3. 原始指標數據**
```bash
# 查看所有患者指標
curl -s http://localhost:8000/metrics | grep patient_

# 查看異常警報
curl -s http://localhost:8000/metrics | grep medical_alerts_total
```

---

## 🎯 **支持的醫療設備**

系統會自動識別以下設備的醫療參數：

| 設備型號 | 監控參數 |
|---------|---------|
| **Philips MP20** | 心率、血氧、血壓、呼吸頻率 |
| **Dräger C500** | FiO2、PEEP、PIP、Pinsp |  
| **SOMANETICS INVOS** | 左右腦氧飽和度 |

---

## 🔔 **異常警報設定**

系統會自動檢測以下異常情況：

| 參數 | 正常範圍 | 警報觸發條件 |
|------|---------|-------------|
| 心率 | 50-150 bpm | < 40 或 > 180 (危急) |
| 血氧 | ≥ 95% | < 90% (危急)，90-95% (警告) |
| 收縮壓 | 90-180 mmHg | < 90 或 > 180 (危急) |
| 舒張壓 | 60-110 mmHg | < 60 或 > 110 (危急) |

---

## 🛠️ **常見問題**

### **問題1：Medical Exporter 無法啟動**
```bash
# 檢查端口是否被佔用
netstat -tuln | grep 8000

# 如果被佔用，終止進程
pkill -f medical_mongodb_exporter

# 重新啟動
python3 medical_mongodb_exporter.py
```

### **問題2：沒有看到醫療數據**
```bash
# 檢查 MongoDB 中的數據
python3 -c "
from mongo_manager import get_mongo_manager
mgr = get_mongo_manager()
mgr.connect_to_mongodb()
count = mgr.db.screen_analysis.count_documents({'success': True})
print(f'醫療數據筆數: {count}')
"
```

### **問題3：Grafana 無法訪問**
```bash
# 檢查服務運行狀態
docker-compose -f docker-compose.monitoring.yml ps

# 重啟 Grafana（如果使用 Docker）
docker-compose -f docker-compose.monitoring.yml restart grafana
```

### **問題4：影片處理失敗**
- 確認 OpenAI API Key 有效
- 檢查影片檔案是否存在
- 確認網路連線正常

---

## 📞 **獲得幫助**

### **系統測試**
```bash
# 執行完整系統測試
./test_medical_monitoring.sh
```

### **查看日誌**
```bash
# Medical Exporter 日誌
tail -f medical_exporter.log

# 影片處理日誌  
tail -f video_processing_test.log
```

### **文檔資源**
- 📋 **完整說明**：[README_MONITORING.md](README_MONITORING.md)
- 🔧 **故障排除**：[TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 💡 **使用技巧**

1. **批量處理影片**：可以寫一個簡單的腳本來處理多個影片檔案
2. **自定義警報**：修改 `alerts/medical_alerts.yml` 來調整警報閾值
3. **添加新設備**：更新 `templates/` 目錄來支持新的醫療設備
4. **數據匯出**：可以直接從 Prometheus 匯出時間序列數據進行分析

---

**🎉 享受使用醫療監控系統！如有問題請參考故障排除指南或執行系統測試。**