import { writeFileSync } from "node:fs";
import { OrderAPI } from "../../../index";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";

describe("临时探针：软删订单复核", () => {
  test("order should be invisible", async () => {
    const lines: string[] = [];
    await login(USERS.ADMIN.username);
    try {
      const detail = await OrderAPI.getDetail("DH20260929194549180349");
      lines.push(`getDetail: VISIBLE status=${(detail as any).status}`);
    } catch (e: any) {
      lines.push(`getDetail: NOT_VISIBLE code=${e?.response?.data?.code ?? e?.message}`);
    }
    await login("test");
    try {
      const page = await OrderAPI.getPage({ pageNum: 1, pageSize: 50 } as any);
      const hit = ((page as any).list ?? []).filter(
        (o: any) => o.orderNo === "DH20260929194549180349"
      );
      lines.push(`user list total=${(page as any).total} hit=${hit.length}`);
    } catch (e: any) {
      lines.push(`list error: ${e?.message}`);
    }
    writeFileSync("/tmp/probe_order_gone.txt", lines.join("\n"));
  });
});
