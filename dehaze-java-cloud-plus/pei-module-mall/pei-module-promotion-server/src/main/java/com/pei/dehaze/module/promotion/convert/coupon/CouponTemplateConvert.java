package com.pei.dehaze.module.promotion.convert.coupon;

import cn.hutool.core.map.MapUtil;
import com.pei.dehaze.framework.common.pojo.PageResult;
import com.pei.dehaze.module.promotion.controller.admin.coupon.vo.template.CouponTemplateCreateReqVO;
import com.pei.dehaze.module.promotion.controller.admin.coupon.vo.template.CouponTemplatePageReqVO;
import com.pei.dehaze.module.promotion.controller.admin.coupon.vo.template.CouponTemplateRespVO;
import com.pei.dehaze.module.promotion.controller.admin.coupon.vo.template.CouponTemplateUpdateReqVO;
import com.pei.dehaze.module.promotion.controller.app.coupon.vo.template.AppCouponTemplatePageReqVO;
import com.pei.dehaze.module.promotion.controller.app.coupon.vo.template.AppCouponTemplateRespVO;
import com.pei.dehaze.module.promotion.dal.dataobject.coupon.CouponTemplateDO;
import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;

import java.util.List;
import java.util.Map;

/**
 * 优惠劵模板 Convert
 *
 * @author earthyzinc
 */
@Mapper
public interface CouponTemplateConvert {

    CouponTemplateConvert INSTANCE = Mappers.getMapper(CouponTemplateConvert.class);

    CouponTemplateDO convert(CouponTemplateCreateReqVO bean);

    CouponTemplateDO convert(CouponTemplateUpdateReqVO bean);

    CouponTemplateRespVO convert(CouponTemplateDO bean);

    PageResult<CouponTemplateRespVO> convertPage(PageResult<CouponTemplateDO> page);

    CouponTemplatePageReqVO convert(AppCouponTemplatePageReqVO pageReqVO, List<Integer> canTakeTypes, Integer productScope, Long productScopeValue);

    default PageResult<AppCouponTemplateRespVO> convertAppPage(PageResult<CouponTemplateDO> pageResult, Map<Long, Boolean> userCanTakeMap) {
        PageResult<AppCouponTemplateRespVO> result = convertAppPage(pageResult);
        copyTo(result.getList(), userCanTakeMap);
        return result;
    }

    PageResult<AppCouponTemplateRespVO> convertAppPage(PageResult<CouponTemplateDO> pageResult);

    default void copyTo(List<AppCouponTemplateRespVO> list, Map<Long, Boolean> userCanTakeMap) {
        for (AppCouponTemplateRespVO template : list) {
            // 检查已领取数量是否超过限领数量
            template.setCanTake(MapUtil.getBool(userCanTakeMap, template.getId(), false));
        }
    }

    default List<AppCouponTemplateRespVO> convertAppList(List<CouponTemplateDO> list, Map<Long, Boolean> userCanTakeMap) {
        List<AppCouponTemplateRespVO> result = convertAppList(list);
        copyTo(result, userCanTakeMap);
        return result;
    }

    List<AppCouponTemplateRespVO> convertAppList(List<CouponTemplateDO> list);

    List<CouponTemplateRespVO> convertList(List<CouponTemplateDO> list);

}
