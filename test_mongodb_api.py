#!/usr/bin/env python3
"""
測試 MongoDB API 功能
需要先重啟 Flask 應用程式才能使用新的路由
"""

import requests
import json

BASE_URL = "http://localhost:3001"

def test_mongodb_status():
    """測試 MongoDB 連接狀態"""
    print("=" * 50)
    print("測試 1: MongoDB 連接狀態")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/api/mongodb/status")
        print(f"狀態碼: {response.status_code}")
        print(f"回應內容:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def test_mongodb_videos():
    """測試取得影片列表"""
    print("\n" + "=" * 50)
    print("測試 2: 取得影片列表")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/api/mongodb/videos")
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"回應內容:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"\n📊 總共 {data['count']} 筆影片記錄")
            return True
        else:
            print(f"錯誤: {response.text}")
            return False
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def test_mongodb_medical_values():
    """測試取得醫療數值（需要先有 session_id）"""
    print("\n" + "=" * 50)
    print("測試 3: 取得醫療數值")
    print("=" * 50)
    
    # 先取得一個 session_id
    try:
        response = requests.get(f"{BASE_URL}/api/mongodb/videos")
        if response.status_code == 200:
            data = response.json()
            if data['count'] > 0:
                session_id = data['data'][0]['session_id']
                print(f"使用 Session ID: {session_id}")
                
                # 測試取得醫療數值
                response = requests.get(f"{BASE_URL}/api/mongodb/medical_values/{session_id}")
                print(f"狀態碼: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"回應內容:")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    return True
                else:
                    print(f"錯誤: {response.text}")
                    return False
            else:
                print("資料庫中沒有影片記錄，跳過此測試")
                return True
        else:
            print(f"無法取得影片列表: {response.text}")
            return False
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def main():
    print("🧪 MongoDB API 功能測試")
    print("=" * 50)
    print("注意: 如果測試失敗，請先重啟 Flask 應用程式")
    print("重啟指令: pkill -f web_app.py && python web_app.py")
    print("=" * 50)
    print()
    
    results = []
    
    # 測試 1: MongoDB 狀態
    results.append(("MongoDB 連接狀態", test_mongodb_status()))
    
    # 測試 2: 影片列表
    results.append(("影片列表 API", test_mongodb_videos()))
    
    # 測試 3: 醫療數值
    results.append(("醫療數值 API", test_mongodb_medical_values()))
    
    # 總結
    print("\n" + "=" * 50)
    print("測試結果總結")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！MongoDB API 功能正常。")
    else:
        print("\n⚠️  部分測試失敗，請檢查 Flask 應用程式是否已重啟。")

if __name__ == "__main__":
    main()