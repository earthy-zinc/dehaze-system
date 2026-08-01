import Redis from "ioredis";
import { DEHAZE_HOST, DEHAZE_PASSWORD } from "#/config/constant";

let redis: Redis | null = null;

export function getRedis(): Redis {
  if (!redis) {
    redis = new Redis({
      host: DEHAZE_HOST,
      port: 6379,
      password: DEHAZE_PASSWORD,
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
