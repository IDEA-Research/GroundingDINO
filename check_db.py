from pymongo import MongoClient
import pprint

# 連線設定 (根據 mongo_manager.py 的預設值)
client = MongoClient('mongodb://localhost:27017')
db = client['medical_monitor_db']

# 查詢影片總數
count = db.video_analysis.count_documents({})
print(f"📊 目前資料庫裡共有 {count} 筆影片分析紀錄")

# 列出最近一筆
if count > 0:
    print("\n📝 最近一筆資料內容：")
    latest = db.video_analysis.find_one(sort=[('timestamp', -1)])
    pprint.pprint(latest)
else:
    print("❌ 資料庫是空的！")