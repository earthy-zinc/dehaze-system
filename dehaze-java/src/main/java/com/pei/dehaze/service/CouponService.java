package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysCoupon;
import com.pei.dehaze.model.form.CouponBatchDistributeForm;
import com.pei.dehaze.model.form.CouponForm;
import com.pei.dehaze.model.query.CouponPageQuery;
import com.pei.dehaze.model.vo.CouponBatchResult;
import com.pei.dehaze.model.vo.CouponCreateResult;
import com.pei.dehaze.model.vo.CouponReceiveResult;
import com.pei.dehaze.model.vo.CouponVO;
import com.pei.dehaze.model.vo.UserCouponVO;

import java.util.List;

public interface CouponService extends IService<SysCoupon> {

    CouponCreateResult create(CouponForm form);

    void update(Long id, CouponForm form);

    void deleteByIds(String ids);

    CouponBatchResult batchDistribute(CouponBatchDistributeForm form);

    CouponReceiveResult receive(Long couponId);

    List<UserCouponVO> listMy(Integer status);

    Page<CouponVO> getPage(CouponPageQuery query);
}
