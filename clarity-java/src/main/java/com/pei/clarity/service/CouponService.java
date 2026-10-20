package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysCoupon;
import com.pei.clarity.model.form.CouponBatchDistributeForm;
import com.pei.clarity.model.form.CouponForm;
import com.pei.clarity.model.query.CouponPageQuery;
import com.pei.clarity.model.vo.CouponBatchResult;
import com.pei.clarity.model.vo.CouponCreateResult;
import com.pei.clarity.model.vo.CouponReceiveResult;
import com.pei.clarity.model.vo.CouponVO;
import com.pei.clarity.model.vo.UserCouponVO;
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
