import logging
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.client: AsyncIOMotorClient | None = None
        self.db = None
        self.collection = None
        self._connected = False

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
            )
            self.db = self.client[settings.DB_NAME]
            self.collection = self.db[settings.COLLECTION_NAME]
            await self.client.admin.command("ping")
            self._connected = True
            logger.info("MongoDB connected")
        except Exception as e:
            self._connected = False
            logger.error("MongoDB connection failed: %s", e)
            raise

    async def disconnect(self):
        if self.client:
            self.client.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def fetch_all(self) -> list[dict]:
        if not self._connected:
            return []
        cursor = self.collection.find({}, {"_id": 0, "period": 1, "result": 1}).sort("_id", 1).limit(settings.MAX_RECORDS)
        return await cursor.to_list(length=None)

    async def fetch_recent(self, limit: int = 500) -> list[dict]:
        if not self._connected:
            return []
        cursor = self.collection.find({}, {"_id": 0, "period": 1, "result": 1}).sort("_id", -1).limit(limit)
        docs = await cursor.to_list(length=None)
        docs.reverse()
        return docs

    async def get_count(self) -> int:
        if not self._connected:
            return 0
        return await self.collection.count_documents({})

    async def get_stats(self) -> dict:
        if not self._connected:
            return {"total": 0, "red": 0, "green": 0}
        total = await self.collection.count_documents({})
        red = await self.collection.count_documents({"result": "Red"})
        green = await self.collection.count_documents({"result": "Green"})
        return {"total": total, "red": red, "green": green}

    async def get_latest_period(self) -> str | None:
        if not self._connected:
            return None
        doc = await self.collection.find_one({}, {"period": 1}, sort=[("_id", -1)])
        return doc["period"] if doc else None

    async def analyze_by_hour(self) -> dict:
        if not self._connected:
            return {}
        pipeline = [
            {
                "$group": {
                    "_id": {"$substrCP": ["$period", 8, 2]},
                    "total": {"$sum": 1},
                    "red": {"$sum": {"$cond": [{"$eq": ["$result", "Red"]}, 1, 0]}},
                    "green": {"$sum": {"$cond": [{"$eq": ["$result", "Green"]}, 1, 0]}},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = self.collection.aggregate(pipeline, allowDiskUse=True)
        result = {}
        async for doc in cursor:
            hour = str(doc["_id"]).zfill(2)
            t = doc["total"]
            r = doc["red"]
            g = doc["green"]
            result[hour] = {
                "total": t,
                "red": r,
                "green": g,
                "red_pct": round(r / t * 100, 1) if t else 0,
                "green_pct": round(g / t * 100, 1) if t else 0,
                "dominant": "Red" if r > g else "Green" if g > r else "Equal",
            }
        return result

    async def analyze_streaks(self) -> dict:
        docs = await self.fetch_recent(500)
        if len(docs) < 2:
            return {"max_streak_red": 0, "max_streak_green": 0, "avg_streak_red": 0, "avg_streak_green": 0}
        streaks = {"Red": [], "Green": []}
        current = docs[0]["result"]
        count = 1
        for doc in docs[1:]:
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


db_manager = DatabaseManager()
