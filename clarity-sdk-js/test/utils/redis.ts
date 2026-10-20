import Redis from "ioredis";
import { REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DATABASE } from "#/config/constant";

let redis: Redis | null = null;

export function getRedis(): Redis {
  if (!redis) {
    redis = new Redis({
      host: REDIS_HOST,
      port: REDIS_PORT,
      password: REDIS_PASSWORD,
      db: REDIS_DATABASE,
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
