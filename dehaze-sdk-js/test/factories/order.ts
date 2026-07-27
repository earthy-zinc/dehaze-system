import { OrderCreateForm, OrderQuery, RefundApplyForm, RefundQuery } from "@/api/order/model";
import { pageQuery, uniqueName } from "./common";

/** 分转换为元 */
const yuan = (v: number) => v * 100;

export function createOrderCreateForm(
  packageId: number,
  overrides: Partial<OrderCreateForm> = {}
): OrderCreateForm {
  return {
    packageId,
    payMethod: "balance",
    ...overrides,
  };
}

export function createOrderQuery(overrides: Partial<OrderQuery> = {}): OrderQuery {
  return pageQuery<OrderQuery>({ ...overrides });
}

export function createRefundApplyForm(overrides: Partial<RefundApplyForm> = {}): RefundApplyForm {
  return {
    reason: "test_refund",
    ...overrides,
  };
}

export function createRefundQuery(overrides: Partial<RefundQuery> = {}): RefundQuery {
  return pageQuery<RefundQuery>({ ...overrides });
}
