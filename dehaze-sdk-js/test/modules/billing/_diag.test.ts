import { AiBillingAPI, AuthAPI } from "../../../index";
import { login } from "#/utils/auth";
import { getRedis } from "#/utils/redis";

describe("diag", () => {
  it("diagnose balance/records", async () => {
    await login("admin");
    for (const [name, p] of [
      ["balance", AiBillingAPI.getBalance()],
      ["records", AiBillingAPI.getRecords({ pageNum: 1, pageSize: 10 })],
      ["credit-logs", AiBillingAPI.getCreditLogs({ pageNum: 1, pageSize: 10 })],
      ["stats", AiBillingAPI.getStats({ groupBy: "model" })],
    ] as const) {
      try {
        const r = await p;
        console.log(`>>> ${name} OK`, JSON.stringify(r).slice(0, 200));
      } catch (e: any) {
        console.log(
          `>>> ${name} ERR status=${e?.response?.status} code=${e?.response?.data?.code} msg=${e?.response?.data?.msg} data=${JSON.stringify(e?.response?.data?.data)}`
        );
      }
    }
    const redis = getRedis();
    console.log("=== quota keys for admin(2) ===");
    for (const k of await redis.keys("ai:quota:*")) console.log(k);
  });
});
