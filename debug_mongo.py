from mongo_manager import get_mongo_manager
import json
from bson import json_util

def check_data():
    mm = get_mongo_manager()
    if not mm.start_mongodb():
        print("MongoDB start failed")
        return
    
    if mm.db is None:
        print("mm.db is None")
        return

    session_id = "6f120944-4888-48a3-915a-380f201cf4eb"
    
    # 1. Check video_analysis
    video = mm.db.video_analysis.find_one({"session_id": session_id})
    print(f"\n--- Video Analysis ({session_id}) ---")
    print(json.dumps(video, indent=2, default=json_util.default))
    
    if video:
        video_id = video["_id"]
        # 2. Check frame_results count
        frames_count = mm.db.frame_results.count_documents({"video_analysis_id": video_id})
        print(f"\nFrame Results Count: {frames_count}")
        
        # 3. Check screen_analysis models and counts
        pipeline = [
            {"$lookup": {
                "from": "frame_results",
                "localField": "frame_result_id",
                "foreignField": "_id",
                "as": "frame"
            }},
            {"$unwind": "$frame"},
            {"$match": {"frame.video_analysis_id": video_id}},
            {"$group": {"_id": "$llm_model", "count": {"$sum": 1}, "success_count": {"$sum": {"$cond": ["$success", 1, 0]}}}}
        ]
        stats = list(mm.db.screen_analysis.aggregate(pipeline))
        print("\n--- Screen Analysis Stats for this Session ---")
        for s in stats:
            print(f"Model: {s['_id']}, Total: {s['count']}, Success: {s['success_count']}")

        # 4. Check a sample medical value
        sample = mm.db.screen_analysis.find_one({
            "frame_result_id": {"$in": [f["_id"] for f in mm.db.frame_results.find({"video_analysis_id": video_id}, {"_id": 1})]}
        })
        print("\n--- Sample Screen Analysis ---")
        print(json.dumps(sample, indent=2, default=json_util.default))

if __name__ == "__main__":
    check_data()
