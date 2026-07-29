package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysMemberBenefitMapper;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.form.BenefitForm;
import com.pei.dehaze.model.vo.BenefitVO;
import com.pei.dehaze.service.MemberBenefitService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MemberBenefitServiceImpl extends ServiceImpl<SysMemberBenefitMapper, SysMemberBenefit> implements MemberBenefitService {

    private final StringRedisTemplate stringRedisTemplate;

    @Override
    public SysMemberBenefit getByLevelCode(String levelCode) {
        return this.getOne(new LambdaQueryWrapper<SysMemberBenefit>()
                .eq(SysMemberBenefit::getLevelCode, levelCode));
    }

    @Override
    public List<SysMemberBenefit> listAllOrdered() {
        return this.list(new LambdaQueryWrapper<SysMemberBenefit>()
                .eq(SysMemberBenefit::getStatus, 1)
                .orderByAsc(SysMemberBenefit::getSort));
    }

    @Override
    public List<BenefitVO> listVOs() {
        List<SysMemberBenefit> benefits = this.list(new LambdaQueryWrapper<SysMemberBenefit>()
                .orderByAsc(SysMemberBenefit::getSort));
        return benefits.stream().map(this::toVO).toList();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateByLevelCode(String levelCode, BenefitForm form) {
        SysMemberBenefit benefit = this.getByLevelCode(levelCode);
        if (benefit == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND, "权益配置不存在");
        }
        if (form.getLevelName() != null) {
            benefit.setLevelName(form.getLevelName());
        }
        if (form.getGrowthMin() != null) {
            benefit.setGrowthMin(form.getGrowthMin());
        }
        if (form.getGrowthMax() != null) {
            benefit.setGrowthMax(form.getGrowthMax());
        }
        if (form.getMonthlyDehazeQuota() != null) {
            benefit.setMonthlyDehazeQuota(form.getMonthlyDehazeQuota());
        }
        if (form.getMonthlyEvaluateQuota() != null) {
            benefit.setMonthlyEvaluateQuota(form.getMonthlyEvaluateQuota());
        }
        if (form.getHistoryRetention() != null) {
            benefit.setHistoryRetention(form.getHistoryRetention());
        }
        if (form.getBatchLimit() != null) {
            benefit.setBatchLimit(form.getBatchLimit());
        }
        if (form.getPriority() != null) {
            benefit.setPriority(form.getPriority());
        }
        if (form.getAdvancedParams() != null) {
            benefit.setAdvancedParams(form.getAdvancedParams());
        }
        if (form.getHdExport() != null) {
            benefit.setHdExport(form.getHdExport());
        }
        if (form.getReportExport() != null) {
            benefit.setReportExport(form.getReportExport());
        }
        if (form.getBatchDownload() != null) {
            benefit.setBatchDownload(form.getBatchDownload());
        }
        if (form.getSort() != null) {
            benefit.setSort(form.getSort());
        }
        if (form.getStatus() != null) {
            benefit.setStatus(form.getStatus());
        }
        if (benefit.getGrowthMax() != null && benefit.getGrowthMax() > 0
                && benefit.getGrowthMin() != null && benefit.getGrowthMin() > benefit.getGrowthMax()) {
            throw new BusinessException(ResultCode.BENEFIT_CONFIG_INVALID, "成长值下限不能大于上限");
        }
        this.updateById(benefit);
        stringRedisTemplate.delete("member:benefit:" + levelCode);
        stringRedisTemplate.delete("member:benefit:all");
    }

    private BenefitVO toVO(SysMemberBenefit benefit) {
        BenefitVO vo = new BenefitVO();
        vo.setLevelCode(benefit.getLevelCode());
        vo.setLevelName(benefit.getLevelName());
        vo.setGrowthMin(benefit.getGrowthMin());
        vo.setGrowthMax(benefit.getGrowthMax());
        vo.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
        vo.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
        vo.setHistoryRetention(benefit.getHistoryRetention());
        vo.setBatchLimit(benefit.getBatchLimit());
        vo.setPriority(benefit.getPriority());
        vo.setAdvancedParams(benefit.getAdvancedParams());
        vo.setHdExport(benefit.getHdExport());
        vo.setReportExport(benefit.getReportExport());
        vo.setBatchDownload(benefit.getBatchDownload());
        vo.setSort(benefit.getSort());
        vo.setStatus(benefit.getStatus());
        return vo;
    }
}
