package com.pei.clarity.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.SysMemberBenefitMapper;
import com.pei.clarity.mapper.SysMemberMapper;
import com.pei.clarity.mapper.SysPackageMapper;
import com.pei.clarity.model.entity.SysMember;
import com.pei.clarity.model.entity.SysMemberBenefit;
import com.pei.clarity.model.entity.SysPackage;
import com.pei.clarity.model.form.BenefitForm;
import com.pei.clarity.model.form.BenefitOverrides;
import com.pei.clarity.model.vo.BenefitVO;
import com.pei.clarity.service.MemberBenefitService;
import java.time.LocalDateTime;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class MemberBenefitServiceImpl extends ServiceImpl<SysMemberBenefitMapper, SysMemberBenefit>
        implements MemberBenefitService {

    /** 配额快照与缓存覆盖的 8 类任务（python QUOTA_TASK_TYPES 同款） */
    private static final List<String> QUOTA_TASK_TYPES =
            List.of("dehaze", "derain", "desnow", "lowlight", "super_resolution", "denoise", "inpaint", "evaluate");

    private final StringRedisTemplate stringRedisTemplate;
    private final SysMemberMapper sysMemberMapper;
    private final SysPackageMapper sysPackageMapper;
    private final ObjectMapper objectMapper;

    @Override
    public SysMemberBenefit getByLevelCode(String levelCode) {
        return this.getOne(new LambdaQueryWrapper<SysMemberBenefit>().eq(SysMemberBenefit::getLevelCode, levelCode));
    }

    @Override
    public List<SysMemberBenefit> listAllOrdered() {
        return this.list(new LambdaQueryWrapper<SysMemberBenefit>()
                .eq(SysMemberBenefit::getStatus, 1)
                .orderByAsc(SysMemberBenefit::getSort));
    }

    @Override
    public List<BenefitVO> listVOs() {
        List<SysMemberBenefit> benefits =
                this.list(new LambdaQueryWrapper<SysMemberBenefit>().orderByAsc(SysMemberBenefit::getSort));
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
        if (form.getMonthlyDerainQuota() != null) {
            benefit.setMonthlyDerainQuota(form.getMonthlyDerainQuota());
        }
        if (form.getMonthlyDesnowQuota() != null) {
            benefit.setMonthlyDesnowQuota(form.getMonthlyDesnowQuota());
        }
        if (form.getMonthlyLowlightQuota() != null) {
            benefit.setMonthlyLowlightQuota(form.getMonthlyLowlightQuota());
        }
        if (form.getMonthlySuperResolutionQuota() != null) {
            benefit.setMonthlySuperResolutionQuota(form.getMonthlySuperResolutionQuota());
        }
        if (form.getMonthlyDenoiseQuota() != null) {
            benefit.setMonthlyDenoiseQuota(form.getMonthlyDenoiseQuota());
        }
        if (form.getMonthlyInpaintQuota() != null) {
            benefit.setMonthlyInpaintQuota(form.getMonthlyInpaintQuota());
        }
        if (form.getMonthlyEvaluateQuota() != null) {
            benefit.setMonthlyEvaluateQuota(form.getMonthlyEvaluateQuota());
        }
        if (form.getAiCreditsDaily() != null) {
            benefit.setAiCreditsDaily(form.getAiCreditsDaily());
        }
        if (form.getAiCreditsMonthly() != null) {
            benefit.setAiCreditsMonthly(form.getAiCreditsMonthly());
        }
        if (form.getMultimodalLimit() != null) {
            benefit.setMultimodalLimit(form.getMultimodalLimit());
        }
        if (form.getVipGiftCredits() != null) {
            benefit.setVipGiftCredits(form.getVipGiftCredits());
        }
        if (form.getHistoryRetention() != null) {
            benefit.setHistoryRetention(form.getHistoryRetention());
        }
        if (form.getBatchLimit() != null) {
            benefit.setBatchLimit(form.getBatchLimit());
        }
        if (form.getMaxDevices() != null) {
            benefit.setMaxDevices(form.getMaxDevices());
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
        if (benefit.getGrowthMax() != null
                && benefit.getGrowthMax() > 0
                && benefit.getGrowthMin() != null
                && benefit.getGrowthMin() > benefit.getGrowthMax()) {
            throw new BusinessException(ResultCode.BENEFIT_CONFIG_INVALID, "成长值下限不能大于上限");
        }
        this.updateById(benefit);
        refreshLevelMemberQuotas(levelCode, benefit);
        stringRedisTemplate.delete("member:benefit:" + levelCode);
        stringRedisTemplate.delete("member:benefit:all");
    }

    /** 权益配置修改立即生效：回刷该等级活跃会员的 8 类配额快照并失效其配额缓存 */
    private void refreshLevelMemberQuotas(String levelCode, SysMemberBenefit benefit) {
        BenefitOverrides cardOverrides = loadCardOverrides(levelCode);
        List<SysMember> members = sysMemberMapper.selectList(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getLevelCode, levelCode)
                .eq(SysMember::getStatus, 1));
        for (SysMember member : members) {
            applyQuotaSnapshot(member, benefit, isCardActive(member) ? cardOverrides : new BenefitOverrides());
            sysMemberMapper.updateById(member);
            invalidateMemberQuotaCache(member.getUserId());
        }
    }

    @Override
    public void refreshMemberQuotaSnapshot(SysMember member) {
        SysMemberBenefit benefit = getByLevelCode(member.getLevelCode());
        if (benefit == null) {
            return;
        }
        applyQuotaSnapshot(
                member,
                benefit,
                isCardActive(member) ? loadCardOverrides(member.getLevelCode()) : new BenefitOverrides());
    }

    @Override
    public void appendQuotaSnapshot(
            LambdaUpdateWrapper<SysMember> wrapper, SysMember member, SysMemberBenefit benefit) {
        BenefitOverrides overrides =
                isCardActive(member) ? loadCardOverrides(member.getLevelCode()) : new BenefitOverrides();
        wrapper.set(
                SysMember::getMonthlyDehazeQuota,
                effectiveQuota(benefit.getMonthlyDehazeQuota(), overrides.getMonthlyDehazeQuota()));
        wrapper.set(
                SysMember::getMonthlyDerainQuota,
                effectiveQuota(benefit.getMonthlyDerainQuota(), overrides.getMonthlyDerainQuota()));
        wrapper.set(
                SysMember::getMonthlyDesnowQuota,
                effectiveQuota(benefit.getMonthlyDesnowQuota(), overrides.getMonthlyDesnowQuota()));
        wrapper.set(
                SysMember::getMonthlyLowlightQuota,
                effectiveQuota(benefit.getMonthlyLowlightQuota(), overrides.getMonthlyLowlightQuota()));
        wrapper.set(
                SysMember::getMonthlySuperResolutionQuota,
                effectiveQuota(benefit.getMonthlySuperResolutionQuota(), overrides.getMonthlySuperResolutionQuota()));
        wrapper.set(
                SysMember::getMonthlyDenoiseQuota,
                effectiveQuota(benefit.getMonthlyDenoiseQuota(), overrides.getMonthlyDenoiseQuota()));
        wrapper.set(
                SysMember::getMonthlyInpaintQuota,
                effectiveQuota(benefit.getMonthlyInpaintQuota(), overrides.getMonthlyInpaintQuota()));
        wrapper.set(
                SysMember::getMonthlyEvaluateQuota,
                effectiveQuota(benefit.getMonthlyEvaluateQuota(), overrides.getMonthlyEvaluateQuota()));
    }

    @Override
    public void invalidateMemberQuotaCache(Long userId) {
        stringRedisTemplate.delete("member:level:" + userId);
        for (String taskType : QUOTA_TASK_TYPES) {
            stringRedisTemplate.delete("member:quota:" + userId + ":" + taskType);
        }
    }

    /** 已购会员卡（level_source=purchase 且未到期）的覆盖项才生效，其余回退等级权益 */
    private boolean isCardActive(SysMember member) {
        return "purchase".equals(member.getLevelSource())
                && (member.getExpireTime() == null || member.getExpireTime().isAfter(LocalDateTime.now()));
    }

    private void applyQuotaSnapshot(SysMember member, SysMemberBenefit benefit, BenefitOverrides overrides) {
        member.setMonthlyDehazeQuota(
                effectiveQuota(benefit.getMonthlyDehazeQuota(), overrides.getMonthlyDehazeQuota()));
        member.setMonthlyDerainQuota(
                effectiveQuota(benefit.getMonthlyDerainQuota(), overrides.getMonthlyDerainQuota()));
        member.setMonthlyDesnowQuota(
                effectiveQuota(benefit.getMonthlyDesnowQuota(), overrides.getMonthlyDesnowQuota()));
        member.setMonthlyLowlightQuota(
                effectiveQuota(benefit.getMonthlyLowlightQuota(), overrides.getMonthlyLowlightQuota()));
        member.setMonthlySuperResolutionQuota(
                effectiveQuota(benefit.getMonthlySuperResolutionQuota(), overrides.getMonthlySuperResolutionQuota()));
        member.setMonthlyDenoiseQuota(
                effectiveQuota(benefit.getMonthlyDenoiseQuota(), overrides.getMonthlyDenoiseQuota()));
        member.setMonthlyInpaintQuota(
                effectiveQuota(benefit.getMonthlyInpaintQuota(), overrides.getMonthlyInpaintQuota()));
        member.setMonthlyEvaluateQuota(
                effectiveQuota(benefit.getMonthlyEvaluateQuota(), overrides.getMonthlyEvaluateQuota()));
    }

    /** 覆盖项与等级权益取较高值，避免回刷把已购会员卡的额度降回等级基线 */
    private Integer effectiveQuota(Integer base, Integer override) {
        return override != null ? Math.max(base, override) : base;
    }

    private BenefitOverrides loadCardOverrides(String levelCode) {
        SysPackage pkg = sysPackageMapper.selectOne(new LambdaQueryWrapper<SysPackage>()
                .eq(SysPackage::getLevelCode, levelCode)
                .last("LIMIT 1"));
        if (pkg == null
                || pkg.getBenefitOverrides() == null
                || pkg.getBenefitOverrides().isBlank()) {
            return new BenefitOverrides();
        }
        try {
            BenefitOverrides overrides = objectMapper.readValue(pkg.getBenefitOverrides(), BenefitOverrides.class);
            return overrides == null ? new BenefitOverrides() : overrides;
        } catch (JsonProcessingException e) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "套餐权益覆盖项解析失败");
        }
    }

    private BenefitVO toVO(SysMemberBenefit benefit) {
        BenefitVO vo = new BenefitVO();
        vo.setLevelCode(benefit.getLevelCode());
        vo.setLevelName(benefit.getLevelName());
        vo.setGrowthMin(benefit.getGrowthMin());
        vo.setGrowthMax(benefit.getGrowthMax());
        vo.setAiCreditsDaily(benefit.getAiCreditsDaily());
        vo.setAiCreditsMonthly(benefit.getAiCreditsMonthly());
        vo.setMultimodalLimit(benefit.getMultimodalLimit());
        vo.setVipGiftCredits(benefit.getVipGiftCredits());
        vo.setHistoryRetention(benefit.getHistoryRetention());
        vo.setBatchLimit(benefit.getBatchLimit());
        vo.setMaxDevices(benefit.getMaxDevices());
        vo.setPriority(benefit.getPriority());
        vo.setAdvancedParams(benefit.getAdvancedParams());
        vo.setHdExport(benefit.getHdExport());
        vo.setReportExport(benefit.getReportExport());
        vo.setBatchDownload(benefit.getBatchDownload());
        vo.setSort(benefit.getSort());
        vo.setStatus(benefit.getStatus());
        vo.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
        vo.setMonthlyDerainQuota(benefit.getMonthlyDerainQuota());
        vo.setMonthlyDesnowQuota(benefit.getMonthlyDesnowQuota());
        vo.setMonthlyLowlightQuota(benefit.getMonthlyLowlightQuota());
        vo.setMonthlySuperResolutionQuota(benefit.getMonthlySuperResolutionQuota());
        vo.setMonthlyDenoiseQuota(benefit.getMonthlyDenoiseQuota());
        vo.setMonthlyInpaintQuota(benefit.getMonthlyInpaintQuota());
        vo.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
        return vo;
    }
}
