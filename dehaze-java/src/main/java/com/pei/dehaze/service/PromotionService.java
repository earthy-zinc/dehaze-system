package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysPromotion;
import com.pei.dehaze.model.form.PromotionForm;
import com.pei.dehaze.model.form.PromotionPackageForm;
import com.pei.dehaze.model.query.PromotionPageQuery;
import com.pei.dehaze.model.vo.PromotionVO;

public interface PromotionService extends IService<SysPromotion> {

    IPage<PromotionVO> getPage(PromotionPageQuery query);

    PromotionVO save(PromotionForm form);

    PromotionVO update(Long id, PromotionForm form);

    PromotionVO updateStatus(Long id, Integer status);

    void delete(Long id);

    void bindPackages(Long id, PromotionPackageForm form);
}
