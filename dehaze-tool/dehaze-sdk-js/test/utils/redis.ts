import Redis from "ioredis";
import { REDIS_CONFIG } from "../config/backend";

let redis: Redis | null = null;

export function getRedis(): Redis {
  if (!redis) {
    redis = new Redis(REDIS_CONFIG);
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
