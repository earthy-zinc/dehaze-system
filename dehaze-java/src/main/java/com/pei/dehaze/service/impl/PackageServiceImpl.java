package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysCouponMapper;
import com.pei.dehaze.mapper.SysOrderMapper;
import com.pei.dehaze.mapper.SysPackageMapper;
import com.pei.dehaze.mapper.SysPromotionMapper;
import com.pei.dehaze.mapper.SysPromotionPackageMapper;
import com.pei.dehaze.mapper.SysUserCouponMapper;
import com.pei.dehaze.model.entity.SysCoupon;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.entity.SysOrder;
import com.pei.dehaze.model.entity.SysPackage;
import com.pei.dehaze.model.entity.SysPromotion;
import com.pei.dehaze.model.entity.SysPromotionPackage;
import com.pei.dehaze.model.entity.SysUserCoupon;
import com.pei.dehaze.model.form.BenefitOverrides;
import com.pei.dehaze.model.form.PackageForm;
import com.pei.dehaze.model.query.PackagePageQuery;
import com.pei.dehaze.model.vo.PackageDetailVO;
import com.pei.dehaze.model.vo.PackagePageVO;
import com.pei.dehaze.model.vo.PriceResult;
import com.pei.dehaze.model.vo.PromotionVO;
import com.pei.dehaze.model.vo.SalesStatsVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.PackageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class PackageServiceImpl extends ServiceImpl<SysPackageMapper, SysPackage> implements PackageService {

    private static final List<String> PERIOD_NAMES = Arrays.asList("monthly", "quarterly", "yearly");
    private static final Map<String, String> PERIOD_LABELS = Map.of("monthly", "月卡", "quarterly", "季卡", "yearly", "年卡");

    private final MemberBenefitService memberBenefitService;
    private final SysOrderMapper orderMapper;
    private final SysPromotionMapper promotionMapper;
    private final SysPromotionPackageMapper promotionPackageMapper;
    private final SysUserCouponMapper userCouponMapper;
    private final SysCouponMapper couponMapper;
    private final ObjectMapper objectMapper;

    @Override
    @Transactional(readOnly = true)
    public List<PackageDetailVO> listOnSale() {
        List<SysPackage> packages = this.list(new LambdaQueryWrapper<SysPackage>()
                .eq(SysPackage::getStatus, 1)
                .orderByAsc(SysPackage::getSort)
                .orderByAsc(SysPackage::getId));
        return packages.stream().map(this::toDetailVO).toList();
    }

    @Override
    @Transactional(readOnly = true)
    public PackageDetailVO getDetail(Long id) {
        SysPackage pkg = this.getById(id);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        if (pkg.getStatus() == null || pkg.getStatus() != 1) {
            throw new BusinessException(ResultCode.PACKAGE_OFF_SHELF);
        }
        return toDetailVO(pkg);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<PackagePageVO> getPage(PackagePageQuery query) {
        Page<SysPackage> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPackage> wrapper = new LambdaQueryWrapper<SysPackage>()
                .like(CharSequenceUtil.isNotBlank(query.getName()), SysPackage::getName, query.getName())
                .eq(CharSequenceUtil.isNotBlank(query.getLevelCode()), SysPackage::getLevelCode, query.getLevelCode())
                .eq(CharSequenceUtil.isNotBlank(query.getPeriod()), SysPackage::getPeriod, query.getPeriod())
                .eq(query.getStatus() != null, SysPackage::getStatus, query.getStatus())
                .ge(query.getStartTime() != null, SysPackage::getCreateTime, query.getStartTime())
                .le(query.getEndTime() != null, SysPackage::getCreateTime, query.getEndTime())
                .orderByAsc(SysPackage::getSort)
                .orderByDesc(SysPackage::getId);
        this.page(page, wrapper);

        Page<PackagePageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toPageVO).toList());
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public PackageForm getForm(Long id) {
        SysPackage pkg = this.getById(id);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        PackageForm form = new PackageForm();
        form.setId(pkg.getId());
        form.setName(pkg.getName());
        form.setLevelCode(pkg.getLevelCode());
        form.setPeriod(pkg.getPeriod());
        form.setPeriodDays(pkg.getPeriodDays());
        form.setOriginalPrice(pkg.getOriginalPrice());
        form.setSalePrice(pkg.getSalePrice());
        form.setDescription(pkg.getDescription());
        form.setBenefitOverrides(parseBenefitOverrides(pkg.getBenefitOverrides()));
        form.setSort(pkg.getSort());
        form.setStatus(pkg.getStatus());
        return form;
    }

    /**
     * 校验安装包名称唯一性（含软删行参与查重，命中即报占用）。
     */
    private void validateNameUnique(String name, Long excludeId) {
        long count = getBaseMapper().countByNameAll(name);
        if (count == 0) {
            return;
        }
        if (excludeId != null) {
            count = getBaseMapper().countByNameAllExcluding(name, excludeId);
        }
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS,
                    "套餐名称已被历史记录占用，无法重复创建");
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(PackageForm form) {
        validatePackageForm(form);
        validateNameUnique(form.getName(), null);
        SysPackage pkg = new SysPackage();
        pkg.setName(form.getName());
        pkg.setLevelCode(form.getLevelCode());
        pkg.setPeriod(form.getPeriod());
        pkg.setPeriodDays(form.getPeriodDays());
        pkg.setOriginalPrice(form.getOriginalPrice());
        pkg.setSalePrice(form.getSalePrice());
        pkg.setDescription(form.getDescription());
        pkg.setBenefitOverrides(serializeBenefitOverrides(form.getBenefitOverrides()));
        pkg.setSort(form.getSort() != null ? form.getSort() : 0);
        pkg.setStatus(form.getStatus() != null ? form.getStatus() : 0);
        pkg.setSalesCount(0L);
        this.save(pkg);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Long id, PackageForm form) {
        SysPackage pkg = this.getById(id);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        validatePackageForm(form);
        if (!pkg.getName().equals(form.getName())) {
            validateNameUnique(form.getName(), id);
        }
        pkg.setName(form.getName());
        pkg.setLevelCode(form.getLevelCode());
        pkg.setPeriod(form.getPeriod());
        pkg.setPeriodDays(form.getPeriodDays());
        pkg.setOriginalPrice(form.getOriginalPrice());
        pkg.setSalePrice(form.getSalePrice());
        pkg.setDescription(form.getDescription());
        pkg.setBenefitOverrides(serializeBenefitOverrides(form.getBenefitOverrides()));
        if (form.getSort() != null) {
            pkg.setSort(form.getSort());
        }
        this.updateById(pkg);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteByIds(String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).filter(s -> !s.isEmpty()).map(Long::parseLong).toList();
        if (idList.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "未指定删除的套餐ID");
        }
        for (Long id : idList) {
            SysPackage pkg = this.getById(id);
            if (pkg == null) {
                throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
            }
            Long orderCount = orderMapper.selectCount(new LambdaQueryWrapper<SysOrder>()
                    .eq(SysOrder::getPackageId, id));
            if (orderCount > 0) {
                throw new BusinessException(ResultCode.PACKAGE_HAS_ORDERS);
            }
            this.removeById(id);
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateStatus(Long id, Integer status) {
        SysPackage pkg = this.getById(id);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        if (status == 0 && !getActivePromotions(id).isEmpty()) {
            throw new BusinessException(ResultCode.PACKAGE_IN_PROMOTION);
        }
        LambdaUpdateWrapper<SysPackage> wrapper = new LambdaUpdateWrapper<SysPackage>()
                .eq(SysPackage::getId, id)
                .set(SysPackage::getStatus, status);
        this.update(wrapper);
    }

    @Override
    @Transactional(readOnly = true)
    public PriceResult calculatePrice(Long packageId, Long userCouponId) {
        SysPackage pkg = this.getById(packageId);
        if (pkg == null) {
            throw new BusinessException(ResultCode.PACKAGE_NOT_FOUND);
        }
        PriceResult result = new PriceResult();
        result.setOriginalPrice(pkg.getOriginalPrice());
        long salePrice = pkg.getSalePrice();
        long discountAmount = calculatePromotionDiscount(pkg);
        result.setDiscountAmount(discountAmount);
        long couponAmount = 0;
        if (userCouponId != null) {
            couponAmount = calculateCouponAmount(userCouponId, pkg, salePrice - discountAmount);
        }
        result.setCouponAmount(couponAmount);
        long payable = salePrice - discountAmount - couponAmount;
        if (payable < 0) {
            payable = 0;
        }
        result.setPayableAmount(payable);
        return result;
    }

    @Override
    public SalesStatsVO getSalesStats() {
        SalesStatsVO stats = new SalesStatsVO();
        List<SysOrder> paidOrders = orderMapper.selectList(new LambdaQueryWrapper<SysOrder>()
                .in(SysOrder::getStatus, 2, 3));
        stats.setTotalSales((long) paidOrders.size());
        stats.setTotalRevenue(paidOrders.stream().mapToLong(o -> o.getPaidAmount() != null ? o.getPaidAmount() : 0).sum());

        List<SysPackage> allPackages = this.list();
        Map<Long, SysPackage> pkgMap = allPackages.stream()
                .collect(Collectors.toMap(SysPackage::getId, p -> p, (a, b) -> a));
        Set<String> levelCodes = allPackages.stream()
                .map(SysPackage::getLevelCode).filter(java.util.Objects::nonNull)
                .collect(Collectors.toSet());
        Map<String, SysMemberBenefit> benefitMap = new HashMap<>();
        for (String code : levelCodes) {
            SysMemberBenefit b = memberBenefitService.getByLevelCode(code);
            if (b != null) {
                benefitMap.put(code, b);
            }
        }

        Map<Long, SalesStatsVO.PackageStatItem> pkgStatsMap = new LinkedHashMap<>();
        Map<String, SalesStatsVO.LevelStatItem> levelStatsMap = new LinkedHashMap<>();
        Map<String, SalesStatsVO.PeriodStatItem> periodStatsMap = new LinkedHashMap<>();
        for (String p : PERIOD_NAMES) {
            SalesStatsVO.PeriodStatItem item = new SalesStatsVO.PeriodStatItem();
            item.setPeriod(p);
            item.setPeriodName(PERIOD_LABELS.get(p));
            item.setSalesCount(0L);
            item.setRevenue(0L);
            periodStatsMap.put(p, item);
        }

        for (SysOrder order : paidOrders) {
            pkgStatsMap.computeIfAbsent(order.getPackageId(), k -> {
                SalesStatsVO.PackageStatItem item = new SalesStatsVO.PackageStatItem();
                item.setPackageId(order.getPackageId());
                SysPackage pkg = pkgMap.get(order.getPackageId());
                item.setPackageName(pkg != null ? pkg.getName() : order.getPackageName());
                item.setSalesCount(0L);
                item.setRevenue(0L);
                return item;
            });
            SalesStatsVO.PackageStatItem pkgItem = pkgStatsMap.get(order.getPackageId());
            pkgItem.setSalesCount(pkgItem.getSalesCount() + 1);
            pkgItem.setRevenue(pkgItem.getRevenue() + (order.getPaidAmount() != null ? order.getPaidAmount() : 0));

            SysPackage pkg = pkgMap.get(order.getPackageId());
            if (pkg != null) {
                levelStatsMap.computeIfAbsent(pkg.getLevelCode(), k -> {
                    SalesStatsVO.LevelStatItem item = new SalesStatsVO.LevelStatItem();
                    item.setLevelCode(pkg.getLevelCode());
                    SysMemberBenefit benefit = benefitMap.get(pkg.getLevelCode());
                    item.setLevelName(benefit != null ? benefit.getLevelName() : pkg.getLevelCode());
                    item.setSalesCount(0L);
                    item.setRevenue(0L);
                    return item;
                });
                SalesStatsVO.LevelStatItem levelItem = levelStatsMap.get(pkg.getLevelCode());
                levelItem.setSalesCount(levelItem.getSalesCount() + 1);
                levelItem.setRevenue(levelItem.getRevenue() + (order.getPaidAmount() != null ? order.getPaidAmount() : 0));

                if (periodStatsMap.containsKey(pkg.getPeriod())) {
                    SalesStatsVO.PeriodStatItem periodItem = periodStatsMap.get(pkg.getPeriod());
                    periodItem.setSalesCount(periodItem.getSalesCount() + 1);
                    periodItem.setRevenue(periodItem.getRevenue() + (order.getPaidAmount() != null ? order.getPaidAmount() : 0));
                }
            }
        }
        stats.setPackageStats(new ArrayList<>(pkgStatsMap.values()));
        stats.setLevelStats(new ArrayList<>(levelStatsMap.values()));
        stats.setPeriodStats(new ArrayList<>(periodStatsMap.values()));

        SalesStatsVO.CouponStatItem couponStats = new SalesStatsVO.CouponStatItem();
        List<SysCoupon> coupons = couponMapper.selectList(null);
        couponStats.setTotalIssued(coupons.stream().mapToLong(c -> c.getIssuedQty() != null ? c.getIssuedQty() : 0).sum());
        couponStats.setTotalUsed(coupons.stream().mapToLong(c -> c.getUsedQty() != null ? c.getUsedQty() : 0).sum());
        couponStats.setUsageRate(couponStats.getTotalIssued() > 0 ? (double) couponStats.getTotalUsed() / couponStats.getTotalIssued() : 0.0);
        stats.setCouponStats(couponStats);
        return stats;
    }

    private long calculatePromotionDiscount(SysPackage pkg) {
        LocalDateTime now = LocalDateTime.now();
        List<SysPromotionPackage> ppList = promotionPackageMapper.selectList(new LambdaQueryWrapper<SysPromotionPackage>()
                .eq(SysPromotionPackage::getPackageId, pkg.getId()));
        if (ppList.isEmpty()) {
            return 0;
        }
        List<Long> promotionIds = ppList.stream().map(SysPromotionPackage::getPromotionId).distinct().toList();
        Map<Long, SysPromotion> promotionMap = promotionMapper.selectBatchIds(promotionIds).stream()
                .collect(Collectors.toMap(SysPromotion::getId, p -> p));
        long maxDiscount = 0;
        for (SysPromotionPackage pp : ppList) {
            SysPromotion promotion = promotionMap.get(pp.getPromotionId());
            if (promotion == null || promotion.getStatus() != 1) {
                continue;
            }
            if (now.isBefore(promotion.getStartTime()) || now.isAfter(promotion.getEndTime())) {
                continue;
            }
            long discount = 0;
            if ("percent".equals(pp.getDiscountType())) {
                discount = pkg.getSalePrice() * pp.getDiscountValue() / 100;
            } else if ("fixed".equals(pp.getDiscountType())) {
                discount = pp.getDiscountValue();
            }
            if (discount > maxDiscount) {
                maxDiscount = discount;
            }
        }
        return maxDiscount;
    }

    private long calculateCouponAmount(Long userCouponId, SysPackage pkg, long afterDiscountPrice) {
        SysUserCoupon userCoupon = userCouponMapper.selectById(userCouponId);
        if (userCoupon == null) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        if (userCoupon.getStatus() != 1 && userCoupon.getStatus() != 4) {
            throw new BusinessException(ResultCode.COUPON_ALREADY_USED);
        }
        Long userId = SecurityUtils.getUserId();
        if (!userCoupon.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        SysCoupon coupon = couponMapper.selectById(userCoupon.getCouponId());
        if (coupon == null) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        if (coupon.getStatus() != 1) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        List<Long> scope = parseJsonToList(coupon.getApplicableScope());
        if (scope != null && !scope.isEmpty() && !scope.contains(pkg.getId())) {
            throw new BusinessException(ResultCode.COUPON_NOT_APPLICABLE);
        }
        if (userCoupon.getExpireTime() != null && userCoupon.getExpireTime().isBefore(LocalDateTime.now())) {
            throw new BusinessException(ResultCode.COUPON_EXPIRED);
        }
        long couponAmount = 0;
        switch (coupon.getType()) {
            case "full_reduction" -> {
                if (coupon.getThreshold() != null && afterDiscountPrice >= coupon.getThreshold()) {
                    couponAmount = coupon.getFaceValue();
                }
            }
            case "discount" -> couponAmount = afterDiscountPrice * (100 - coupon.getFaceValue()) / 100;
            case "no_threshold", "trial" -> couponAmount = coupon.getFaceValue();
        }
        if (couponAmount > afterDiscountPrice) {
            couponAmount = afterDiscountPrice;
        }
        return couponAmount;
    }

    private void validatePackageForm(PackageForm form) {
        if (form.getSalePrice() > form.getOriginalPrice()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "促销价不能高于原价");
        }
        if (!PERIOD_NAMES.contains(form.getPeriod())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "计费周期非法");
        }
    }

    private PackagePageVO toPageVO(SysPackage pkg) {
        PackagePageVO vo = new PackagePageVO();
        vo.setId(pkg.getId());
        vo.setName(pkg.getName());
        vo.setLevelCode(pkg.getLevelCode());
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(pkg.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : pkg.getLevelCode());
        vo.setPeriod(pkg.getPeriod());
        vo.setPeriodDays(pkg.getPeriodDays());
        vo.setOriginalPrice(pkg.getOriginalPrice());
        vo.setSalePrice(pkg.getSalePrice());
        vo.setDailyPrice(pkg.getPeriodDays() != null && pkg.getPeriodDays() > 0 ? (2 * pkg.getSalePrice() + pkg.getPeriodDays()) / (2 * pkg.getPeriodDays()) : 0);
        vo.setSalesCount(pkg.getSalesCount());
        vo.setStatus(pkg.getStatus());
        vo.setCreateTime(pkg.getCreateTime());
        return vo;
    }

    private PackageDetailVO toDetailVO(SysPackage pkg) {
        PackageDetailVO vo = new PackageDetailVO();
        vo.setId(pkg.getId());
        vo.setName(pkg.getName());
        vo.setLevelCode(pkg.getLevelCode());
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(pkg.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : pkg.getLevelCode());
        vo.setPeriod(pkg.getPeriod());
        vo.setPeriodDays(pkg.getPeriodDays());
        vo.setOriginalPrice(pkg.getOriginalPrice());
        vo.setSalePrice(pkg.getSalePrice());
        vo.setDailyPrice(pkg.getPeriodDays() != null && pkg.getPeriodDays() > 0 ? (2 * pkg.getSalePrice() + pkg.getPeriodDays()) / (2 * pkg.getPeriodDays()) : 0);
        vo.setDescription(pkg.getDescription());
        vo.setSalesCount(pkg.getSalesCount());
        vo.setBenefits(buildBenefits(benefit, parseBenefitOverrides(pkg.getBenefitOverrides())));
        vo.setActivePromotions(getActivePromotions(pkg.getId()));
        return vo;
    }

    private Map<String, Integer> buildBenefits(SysMemberBenefit benefit, BenefitOverrides overrides) {
        Map<String, Integer> benefits = new LinkedHashMap<>();
        if (benefit != null) {
            benefits.put("monthlyDehazeQuota", benefit.getMonthlyDehazeQuota());
            benefits.put("monthlyEvaluateQuota", benefit.getMonthlyEvaluateQuota());
            benefits.put("historyRetention", benefit.getHistoryRetention());
            benefits.put("batchLimit", benefit.getBatchLimit());
            benefits.put("priority", benefit.getPriority());
            benefits.put("advancedParams", benefit.getAdvancedParams());
            benefits.put("hdExport", benefit.getHdExport());
            benefits.put("reportExport", benefit.getReportExport());
            benefits.put("batchDownload", benefit.getBatchDownload());
        }
        if (overrides != null) {
            Map<String, Integer> overridesMap = objectMapper.convertValue(overrides,
                    new TypeReference<Map<String, Integer>>() {});
            benefits.putAll(overridesMap);
        }
        return benefits;
    }

    private List<PromotionVO> getActivePromotions(Long packageId) {
        LocalDateTime now = LocalDateTime.now();
        List<SysPromotionPackage> ppList = promotionPackageMapper.selectList(new LambdaQueryWrapper<SysPromotionPackage>()
                .eq(SysPromotionPackage::getPackageId, packageId));
        if (ppList.isEmpty()) {
            return Collections.emptyList();
        }
        List<Long> promotionIds = ppList.stream().map(SysPromotionPackage::getPromotionId).distinct().toList();
        Map<Long, SysPromotion> promotionMap = promotionMapper.selectBatchIds(promotionIds).stream()
                .collect(Collectors.toMap(SysPromotion::getId, p -> p));
        List<PromotionVO> result = new ArrayList<>();
        for (SysPromotionPackage pp : ppList) {
            SysPromotion promotion = promotionMap.get(pp.getPromotionId());
            if (promotion == null || promotion.getStatus() != 1) {
                continue;
            }
            if (now.isBefore(promotion.getStartTime()) || now.isAfter(promotion.getEndTime())) {
                continue;
            }
            result.add(toPromotionVO(promotion));
        }
        return result;
    }

    private PromotionVO toPromotionVO(SysPromotion promotion) {
        PromotionVO vo = new PromotionVO();
        vo.setId(promotion.getId());
        vo.setName(promotion.getName());
        vo.setType(promotion.getType());
        vo.setDescription(promotion.getDescription());
        vo.setStartTime(promotion.getStartTime());
        vo.setEndTime(promotion.getEndTime());
        vo.setActivityRules(parseJsonToObjectMap(promotion.getActivityRules()));
        vo.setNewUserOnly(promotion.getNewUserOnly());
        vo.setStatus(promotion.getStatus());
        return vo;
    }

    private BenefitOverrides parseBenefitOverrides(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, BenefitOverrides.class);
        } catch (JsonProcessingException e) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "权益配置JSON解析失败");
        }
    }

    private Map<String, Object> parseJsonToObjectMap(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析JSON Object Map失败: {}", json, e);
            return null;
        }
    }

    private List<Long> parseJsonToList(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<Long>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析JSON List失败: {}", json, e);
            return null;
        }
    }

    private String serializeBenefitOverrides(BenefitOverrides overrides) {
        if (overrides == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(overrides);
        } catch (JsonProcessingException e) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "权益配置JSON序列化失败");
        }
    }
}
