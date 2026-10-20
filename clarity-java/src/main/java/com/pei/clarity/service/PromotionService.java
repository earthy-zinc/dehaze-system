package com.pei.clarity.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysPromotion;
import com.pei.clarity.model.form.PromotionForm;
import com.pei.clarity.model.form.PromotionPackageForm;
import com.pei.clarity.model.query.PromotionPageQuery;
import com.pei.clarity.model.vo.PromotionVO;

public interface PromotionService extends IService<SysPromotion> {

    IPage<PromotionVO> getPage(PromotionPageQuery query);

    PromotionVO save(PromotionForm form);

    PromotionVO update(Long id, PromotionForm form);

    PromotionVO updateStatus(Long id, Integer status);

    void delete(Long id);

    void bindPackages(Long id, PromotionPackageForm form);
}
