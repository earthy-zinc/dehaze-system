import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  CouponBatchDistributeForm,
  CouponForm,
  CouponQuery,
  CouponVO,
  PackageDetailVO,
  PackageForm,
  PackageOnSaleVO,
  PackagePageVO,
  PackageQuery,
  PriceResult,
  PromotionForm,
  PromotionPackageForm,
  PromotionQuery,
  PromotionStatus,
  PromotionVO,
  SalesStatsVO,
  UserCouponVO,
} from "./model";

class PackageAPI {
  /** 用户端：在售套餐列表 */
  static listOnSale() {
    return request<PackageOnSaleVO[]>({
      url: "/api/v1/packages",
      method: "get",
    });
  }

  /** 用户端：套餐详情 */
  static getDetail(id: number) {
    return request<PackageDetailVO>({
      url: "/api/v1/packages/" + id,
      method: "get",
    });
  }

  /** 价格计算（下单前预览） */
  static calculatePrice(packageId: number, userCouponId?: number) {
    return request<PriceResult>({
      url: "/api/v1/packages/calculate-price",
      method: "get",
      params: { packageId, userCouponId },
    });
  }

  /** 后台：套餐分页列表 */
  static getPage(queryParams: PackageQuery) {
    return request<PageResult<PackagePageVO[]>>({
      url: "/api/v1/packages/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：获取套餐表单数据 */
  static getForm(id: number) {
    return request<PackageForm>({
      url: "/api/v1/packages/" + id + "/form",
      method: "get",
    });
  }

  /** 后台：新增套餐 */
  static add(data: PackageForm) {
    return request({
      url: "/api/v1/packages",
      method: "post",
      data,
    });
  }

  /** 后台：修改套餐 */
  static update(id: number, data: PackageForm) {
    return request({
      url: "/api/v1/packages/" + id,
      method: "put",
      data,
    });
  }

  /** 后台：上架/下架 */
  static updateStatus(id: number, status: number) {
    return request({
      url: "/api/v1/packages/" + id + "/status",
      method: "put",
      params: { status },
    });
  }

  /** 后台：删除套餐（路径参数，逗号分隔） */
  static deleteByIds(ids: string) {
    if (!ids || !ids.trim()) {
      return Promise.reject(new Error("待删除的套餐 ID 列表不能为空"));
    }
    return request({
      url: "/api/v1/packages/" + ids,
      method: "delete",
    });
  }

  /** 后台：销售统计 */
  static getSalesStats() {
    return request<SalesStatsVO>({
      url: "/api/v1/packages/sales/stats",
      method: "get",
    });
  }
}

class CouponAPI {
  /** 用户端：我的优惠券列表 */
  static listMy(status?: number) {
    return request<UserCouponVO[]>({
      url: "/api/v1/packages/coupons/my",
      method: "get",
      params: status !== undefined ? { status } : undefined,
    });
  }

  /** 用户端：领取优惠券 */
  static receive(couponId: number) {
    return request<{ userCouponId: number }>({
      url: "/api/v1/packages/coupons/" + couponId + "/receive",
      method: "post",
    });
  }

  /** 后台：优惠券分页列表 */
  static getPage(queryParams: CouponQuery) {
    return request<PageResult<CouponVO[]>>({
      url: "/api/v1/packages/coupons/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：创建优惠券 */
  static add(data: CouponForm) {
    return request<{ id: number }>({
      url: "/api/v1/packages/coupons",
      method: "post",
      data,
    });
  }

  /** 后台：批量发放优惠券 */
  static batchDistribute(data: CouponBatchDistributeForm) {
    return request<{ successCount: number; failCount: number }>({
      url: "/api/v1/packages/coupons/batch",
      method: "post",
      data,
    });
  }

  /** 后台：修改优惠券 */
  static update(id: number, data: CouponForm) {
    return request({
      url: "/api/v1/packages/coupons/" + id,
      method: "put",
      data,
    });
  }

  /** 后台：删除优惠券（路径参数，逗号分隔） */
  static deleteByIds(ids: string) {
    if (!ids || !ids.trim()) {
      return Promise.reject(new Error("待删除的优惠券 ID 列表不能为空"));
    }
    return request({
      url: "/api/v1/packages/coupons/" + ids,
      method: "delete",
    });
  }
}

export default PackageAPI;
export { CouponAPI, PromotionAPI };

class PromotionAPI {
  /** 后台：促销活动分页列表 */
  static getPage(queryParams: PromotionQuery) {
    return request<PageResult<PromotionVO[]>>({
      url: "/api/v1/packages/promotions/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：创建促销活动 */
  static add(data: PromotionForm) {
    return request<{ id: number }>({
      url: "/api/v1/packages/promotions",
      method: "post",
      data,
    });
  }

  /** 后台：修改促销活动 */
  static update(id: number, data: PromotionForm) {
    return request({
      url: "/api/v1/packages/promotions/" + id,
      method: "put",
      data,
    });
  }

  /** 后台：促销活动上架/下架 */
  static updateStatus(id: number, status: PromotionStatus) {
    return request({
      url: `/api/v1/packages/promotions/${id}/status`,
      method: "put",
      params: { status },
    });
  }

  /** 后台：删除促销活动 */
  static delete(id: number) {
    return request({
      url: "/api/v1/packages/promotions/" + id,
      method: "delete",
    });
  }

  /** 后台：关联套餐 */
  static bindPackages(id: number, data: PromotionPackageForm) {
    return request({
      url: `/api/v1/packages/promotions/${id}/packages`,
      method: "put",
      data,
    });
  }
}
