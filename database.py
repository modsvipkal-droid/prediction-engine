from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings


class DatabaseManager:
    def __init__(self):
        self.client: AsyncIOMotorClient | None = None
        self.db = None
        self.collection = None

    async def connect(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URI)
        self.db = self.client[settings.DB_NAME]
        self.collection = self.db[settings.COLLECTION_NAME]

    async def disconnect(self):
        if self.client:
            self.client.close()

    async def fetch_all(self) -> list[dict]:
        cursor = self.collection.find({}).sort("_id", 1)
        return await cursor.to_list(length=None)

    async def fetch_recent(self, limit: int = 100) -> list[dict]:
        cursor = self.collection.find({}).sort("_id", -1).limit(limit)
        docs = await cursor.to_list(length=None)
        docs.reverse()
        return docs

    async def get_count(self) -> int:
        return await self.collection.count_documents({})

    async def get_stats(self) -> dict:
        total = await self.collection.count_documents({})
        red = await self.collection.count_documents({"result": "Red"})
        green = await self.collection.count_documents({"result": "Green"})
        return {"total": total, "red": red, "green": green}

    async def get_latest_period(self) -> str | None:
        doc = await self.collection.find_one({}, sort=[("_id", -1)])
        return doc["period"] if doc else None

    async def analyze_by_hour(self) -> dict:
        pipeline = [
            {
                "$group": {
                    "_id": {"$substr": ["$period", 8, 2]},
                    "total": {"$sum": 1},
                    "red": {
                        "$sum": {"$cond": [{"$eq": ["$result", "Red"]}, 1, 0]}
                    },
                    "green": {
                        "$sum": {"$cond": [{"$eq": ["$result", "Green"]}, 1, 0]}
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = self.collection.aggregate(pipeline)
        result = {}
        async for doc in cursor:
            hour = int(doc["_id"])
            total = doc["total"]
            r = doc["red"]
            g = doc["green"]
            result[str(hour).zfill(2)] = {
                "total": total,
                "red": r,
                "green": g,
                "red_pct": round(r / total * 100, 1) if total else 0,
                "green_pct": round(g / total * 100, 1) if total else 0,
                "dominant": "Red" if r > g else "Green" if g > r else "Equal",
            }
        return result

    async def analyze_streaks(self) -> dict:
        all_docs = await self.fetch_recent(500)
        if not all_docs:
            return {}
        streaks = {"Red": [], "Green": []}
        current = all_docs[0]["result"]
        count = 1
        for doc in all_docs[1:]:
            if doc["result"] == current:
                count += 1
            else:
                streaks[current].append(count)
                current = doc["result"]
                count = 1
        streaks[current].append(count)
        return {
            "max_streak_red": max(streaks["Red"]) if streaks["Red"] else 0,
            "max_streak_green": max(streaks["Green"]) if streaks["Green"] else 0,
            "avg_streak_red": round(sum(streaks["Red"]) / len(streaks["Red"]), 1) if streaks["Red"] else 0,
            "avg_streak_green": round(sum(streaks["Green"]) / len(streaks["Green"]), 1) if streaks["Green"] else 0,
        }

    async def analyze_pattern_by_position(self) -> dict:
        all_docs = await self.fetch_recent(200)
        position_map = {}
        for i, doc in enumerate(all_docs):
            pos = i % 10
            if pos not in position_map:
                position_map[pos] = {"Red": 0, "Green": 0}
            position_map[pos][doc["result"]] += 1
        result = {}
        for pos, counts in position_map.items():
            total = counts["Red"] + counts["Green"]
            result[str(pos)] = {
                "red": counts["Red"],
                "green": counts["Green"],
                "dominant": "Red" if counts["Red"] > counts["Green"] else "Green",
                "red_pct": round(counts["Red"] / total * 100, 1) if total else 0,
            }
        return result


db_manager = DatabaseManager()
