import { CouponForm, PackageForm, PackageQuery, CouponQuery } from "@/api/package/model";
import { pageQuery, uniqueName } from "./common";

/** 分转换为元 */
const yuan = (v: number) => Math.round(v * 100);

export function createPackageForm(overrides: Partial<PackageForm> = {}): PackageForm {
  return {
    name: uniqueName("测试套餐"),
    levelCode: "level_1",
    period: "monthly",
    periodDays: 30,
    originalPrice: yuan(19.9),
    salePrice: yuan(9.9),
    sort: 100,
    status: 0,
    ...overrides,
  };
}

export function createPackageQuery(overrides: Partial<PackageQuery> = {}): PackageQuery {
  return pageQuery<PackageQuery>({ ...overrides });
}

export function createCouponForm(overrides: Partial<CouponForm> = {}): CouponForm {
  return {
    name: uniqueName("测试优惠券"),
    type: "full_reduction",
    faceValue: yuan(10),
    threshold: yuan(50),
    validType: "fixed",
    validStart: "2026-01-01 00:00:00",
    validEnd: "2026-12-31 23:59:59",
    totalQty: 100,
    perUserLimit: 1,
    status: 1,
    ...overrides,
  };
}

export function createCouponQuery(overrides: Partial<CouponQuery> = {}): CouponQuery {
  return pageQuery<CouponQuery>({ ...overrides });
}
