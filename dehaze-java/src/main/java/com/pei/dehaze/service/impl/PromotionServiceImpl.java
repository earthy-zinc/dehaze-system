package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysPromotionMapper;
import com.pei.dehaze.mapper.SysPromotionPackageMapper;
import com.pei.dehaze.model.entity.SysPromotion;
import com.pei.dehaze.model.entity.SysPromotionPackage;
import com.pei.dehaze.model.form.PromotionForm;
import com.pei.dehaze.model.form.PromotionPackageForm;
import com.pei.dehaze.model.query.PromotionPageQuery;
import com.pei.dehaze.model.vo.PromotionVO;
import com.pei.dehaze.service.PromotionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class PromotionServiceImpl extends ServiceImpl<SysPromotionMapper, SysPromotion> implements PromotionService {

    private final SysPromotionPackageMapper promotionPackageMapper;
    private final ObjectMapper objectMapper;

    @Override
    public IPage<PromotionVO> getPage(PromotionPageQuery query) {
        Page<SysPromotion> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPromotion> wrapper = new LambdaQueryWrapper<SysPromotion>()
                .like(CharSequenceUtil.isNotBlank(query.getName()), SysPromotion::getName, query.getName())
                .eq(CharSequenceUtil.isNotBlank(query.getType()), SysPromotion::getType, query.getType())
                .eq(query.getStatus() != null, SysPromotion::getStatus, query.getStatus())
                .ge(query.getStartTime() != null, SysPromotion::getStartTime, query.getStartTime())
                .le(query.getEndTime() != null, SysPromotion::getEndTime, query.getEndTime())
                .orderByDesc(SysPromotion::getId);
        this.page(page, wrapper);

        Page<PromotionVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toVO).toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PromotionVO save(PromotionForm form) {
        SysPromotion promotion = new SysPromotion();
        applyForm(promotion, form);
        promotion.setStatus(form.getStatus() != null ? form.getStatus() : 0);
        this.save(promotion);
        return toVO(promotion);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PromotionVO update(Long id, PromotionForm form) {
        SysPromotion promotion = this.getById(id);
        if (promotion == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在");
        }
        applyForm(promotion, form);
        this.updateById(promotion);
        return toVO(promotion);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public PromotionVO updateStatus(Long id, Integer status) {
        SysPromotion promotion = this.getById(id);
        if (promotion == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在");
        }
        promotion.setStatus(status);
        this.updateById(promotion);
        return toVO(promotion);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void delete(Long id) {
        SysPromotion promotion = this.getById(id);
        if (promotion == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在");
        }
        this.removeById(id);
        promotionPackageMapper.delete(new LambdaQueryWrapper<SysPromotionPackage>()
                .eq(SysPromotionPackage::getPromotionId, id));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void bindPackages(Long id, PromotionPackageForm form) {
        SysPromotion promotion = this.getById(id);
        if (promotion == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在");
        }
        // 折扣方式/折扣值取自活动规则，与价格计算口径一致
        Map<String, Object> rules = parseRules(promotion.getActivityRules());
        String ruleType = rules != null ? (String) rules.get("discount_type") : null;
        String discountType = "percent".equals(ruleType) || "fixed".equals(ruleType) || "full_reduction".equals(ruleType)
                ? ruleType
                : ("full_reduction".equals(promotion.getType()) ? "full_reduction" : "percent");
        long ruleValue = 0;
        if (rules != null && rules.get("discount_value") != null) {
            ruleValue = Long.parseLong(String.valueOf(rules.get("discount_value")));
        }
        final String type = discountType;
        final long discountValue = ruleValue;
        List<SysPromotionPackage> packages = form.getPackageIds().stream().map(packageId -> {
            SysPromotionPackage pp = new SysPromotionPackage();
            pp.setPromotionId(id);
            pp.setPackageId(packageId);
            pp.setDiscountType(type);
            pp.setDiscountValue(discountValue);
            return pp;
        }).toList();
        promotionPackageMapper.delete(new LambdaQueryWrapper<SysPromotionPackage>()
                .eq(SysPromotionPackage::getPromotionId, id));
        packages.forEach(promotionPackageMapper::insert);
    }

    private void applyForm(SysPromotion promotion, PromotionForm form) {
        promotion.setName(form.getName());
        promotion.setType(form.getType());
        promotion.setDescription(form.getDescription());
        promotion.setStartTime(form.getStartTime());
        promotion.setEndTime(form.getEndTime());
        promotion.setActivityRules(serializeMapToJson(form.getActivityRules()));
        promotion.setNewUserOnly(form.getNewUserOnly() != null ? form.getNewUserOnly() : 0);
    }

    private PromotionVO toVO(SysPromotion promotion) {
        PromotionVO vo = new PromotionVO();
        vo.setId(promotion.getId());
        vo.setName(promotion.getName());
        vo.setType(promotion.getType());
        vo.setDescription(promotion.getDescription());
        vo.setStartTime(promotion.getStartTime());
        vo.setEndTime(promotion.getEndTime());
        vo.setActivityRules(parseRules(promotion.getActivityRules()));
        vo.setNewUserOnly(promotion.getNewUserOnly());
        vo.setStatus(promotion.getStatus());
        vo.setCreateTime(promotion.getCreateTime());
        return vo;
    }

    private Map<String, Object> parseRules(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析促销活动规则JSON失败: {}", json, e);
            return null;
        }
    }

    private String serializeMapToJson(Map<String, Object> value) {
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "活动规则JSON序列化失败");
        }
    }
}
