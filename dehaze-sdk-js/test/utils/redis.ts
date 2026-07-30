import Redis from "ioredis";

let redis: Redis | null = null;

export function getRedis(): Redis {
  if (!redis) {
    redis = new Redis({
      host: process.env.DEHAZE_HOST || "127.0.0.1",
      port: 6379,
      password: process.env.DEHAZE_PASSWORD || "Dehaze@2026",
      db: 0,
      maxRetriesPerRequest: 3,
    });
    redis.on("error", () => {});
  }
  return redis;
}

export async function disconnectRedis(): Promise<void> {
  if (redis) {
    await redis.quit();
    redis = null;
  }
}
