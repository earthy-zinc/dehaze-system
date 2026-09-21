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
        // 在售列表不返回进行中促销（python 契约：仅详情返回 activePromotions）
        return packages.stream().map(pkg -> toDetailVO(pkg, false)).toList();
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
        return toDetailVO(pkg, true);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<PackagePageVO> getPage(PackagePageQuery query) {
        Page<SysPackage> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPackage> wrapper = new LambdaQueryWrapper<SysPackage>()
                .like(CharSequenceUtil.isNotBlank(query.getName()), SysPackage::getName, query.getName())
                .eq(CharSequenceUtil.isNotBlank(query.getPackageType()), SysPackage::getPackageType, query.getPackageType())
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
        form.setPackageType(pkg.getPackageType());
        form.setLevelCode(pkg.getLevelCode());
        form.setPeriod(pkg.getPeriod());
        form.setPeriodDays(pkg.getPeriodDays());
        form.setCreditAmount(pkg.getCreditAmount());
        form.setOriginalPrice(pkg.getOriginalPrice());
        form.setSalePrice(pkg.getSalePrice());
        form.setDescription(pkg.getDescription());
        form.setBenefitOverrides(parseBenefitOverrides(pkg.getBenefitOverrides()));
        form.setSort(pkg.getSort());
        form.setStatus(pkg.getStatus());
        return form;
    }

    /**
     * 校验安装包名称唯一性（仅活跃行；唯一键含 deleted，软删行不占键位）。
     */
    private void validateNameUnique(String name, Long excludeId) {
        long count = getBaseMapper().selectCount(new LambdaQueryWrapper<SysPackage>()
                .eq(SysPackage::getName, name)
                .ne(excludeId != null, SysPackage::getId, excludeId));
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "套餐名称已存在");
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(PackageForm form) {
        validatePackageForm(form, form.getPackageType());
        validateNameUnique(form.getName(), null);
        SysPackage pkg = new SysPackage();
        pkg.setName(form.getName());
        pkg.setPackageType(form.getPackageType());
        applyTypedFields(pkg, form.getPackageType(), form);
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
        // 商品类型创建后锁定，以库中值为准，请求携带的 packageType 被忽略
        validatePackageForm(form, pkg.getPackageType());
        if (!pkg.getName().equals(form.getName())) {
            validateNameUnique(form.getName(), id);
        }
        pkg.setName(form.getName());
        applyTypedFields(pkg, pkg.getPackageType(), form);
        pkg.setOriginalPrice(form.getOriginalPrice());
        pkg.setSalePrice(form.getSalePrice());
        pkg.setDescription(form.getDescription());
        pkg.setBenefitOverrides(serializeBenefitOverrides(form.getBenefitOverrides()));
        if (form.getSort() != null) {
            pkg.setSort(form.getSort());
        }
        this.updateById(pkg);
    }

    /** 按商品类型差异化写入：会员卡等级/周期/有效期，积分卡可得积分，另一类字段置空 */
    private void applyTypedFields(SysPackage pkg, String packageType, PackageForm form) {
        if ("credit".equals(packageType)) {
            pkg.setCreditAmount(form.getCreditAmount());
            pkg.setLevelCode(null);
            pkg.setPeriod(null);
            pkg.setPeriodDays(null);
        } else {
            pkg.setCreditAmount(null);
            pkg.setLevelCode(form.getLevelCode());
            pkg.setPeriod(form.getPeriod());
            pkg.setPeriodDays(form.getPeriodDays());
        }
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
        // 新用户专享校验：关联 new_user 活动时，有历史付费订单的用户不可购买
        boolean newUserOnly = getActivePromotions(packageId).stream()
                .anyMatch(p -> p.getNewUserOnly() != null && p.getNewUserOnly() == 1);
        if (newUserOnly) {
            Long paidCount = orderMapper.selectCount(new LambdaQueryWrapper<SysOrder>()
                    .eq(SysOrder::getUserId, SecurityUtils.getUserId())
                    .in(SysOrder::getStatus, 2, 3));
            if (paidCount > 0) {
                throw new BusinessException(ResultCode.BUSINESS_ERROR, "该套餐仅限新用户购买");
            }
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
            } else if ("full_reduction".equals(pp.getDiscountType())) {
                // 满减按活动规则 tiers 阶梯匹配：取满足门槛档位中的最大面值
                Map<String, Object> rules = parseRulesToMap(promotion.getActivityRules());
                if (rules != null && rules.get("tiers") instanceof List<?> tiers) {
                    for (Object t : tiers) {
                        if (t instanceof Map<?, ?> tier) {
                            long threshold = tier.get("threshold") instanceof Number tn ? tn.longValue() : 0;
                            long faceValue = tier.get("faceValue") instanceof Number fn ? fn.longValue() : 0;
                            if (pkg.getSalePrice() >= threshold && faceValue > discount) {
                                discount = faceValue;
                            }
                        }
                    }
                }
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
        List<Object> scope = parseJsonToList(coupon.getApplicableScope());
        if (scope != null && !scope.isEmpty()) {
            boolean applicable = false;
            for (Object s : scope) {
                if (s instanceof Number n && n.longValue() == pkg.getId()) {
                    applicable = true;
                    break;
                }
                if (s instanceof String str && str.equals(pkg.getPackageType())) {
                    applicable = true;
                    break;
                }
            }
            if (!applicable) {
                throw new BusinessException(ResultCode.COUPON_NOT_APPLICABLE);
            }
        }
        // 体验券直接激活会员卡权益、不产生订单，不参与下单价格计算
        if ("trial".equals(coupon.getType())) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "体验券不参与价格计算，请通过激活流程使用");
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
            case "no_threshold" -> couponAmount = coupon.getFaceValue();
        }
        if (couponAmount > afterDiscountPrice) {
            couponAmount = afterDiscountPrice;
        }
        return couponAmount;
    }

    private void validatePackageForm(PackageForm form, String packageType) {
        if (form.getSalePrice() > form.getOriginalPrice()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "促销价不能高于原价");
        }
        if ("credit".equals(packageType)) {
            if (form.getCreditAmount() == null || form.getCreditAmount() <= 0) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "积分卡可得积分必须大于0");
            }
        } else {
            if (CharSequenceUtil.isBlank(form.getLevelCode())
                    || CharSequenceUtil.isBlank(form.getPeriod())
                    || form.getPeriodDays() == null) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "会员卡必须设置等级/周期/有效期");
            }
            if (!PERIOD_NAMES.contains(form.getPeriod())) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "计费周期非法");
            }
        }
    }

    private PackagePageVO toPageVO(SysPackage pkg) {
        PackagePageVO vo = new PackagePageVO();
        vo.setId(pkg.getId());
        vo.setName(pkg.getName());
        vo.setPackageType(pkg.getPackageType());
        vo.setLevelCode(pkg.getLevelCode());
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(pkg.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : pkg.getLevelCode());
        vo.setPeriod(pkg.getPeriod());
        vo.setPeriodDays(pkg.getPeriodDays());
        vo.setCreditAmount(pkg.getCreditAmount());
        vo.setOriginalPrice(pkg.getOriginalPrice());
        vo.setSalePrice(pkg.getSalePrice());
        vo.setDailyPrice(pkg.getPeriodDays() != null && pkg.getPeriodDays() > 0 ? (2 * pkg.getSalePrice() + pkg.getPeriodDays()) / (2 * pkg.getPeriodDays()) : 0);
        vo.setCreditUnitPrice(pkg.getCreditAmount() != null && pkg.getCreditAmount() > 0 ? pkg.getSalePrice() / pkg.getCreditAmount() : 0);
        vo.setSalesCount(pkg.getSalesCount());
        vo.setStatus(pkg.getStatus());
        vo.setCreateTime(pkg.getCreateTime());
        return vo;
    }

    private PackageDetailVO toDetailVO(SysPackage pkg, boolean withPromotions) {
        PackageDetailVO vo = new PackageDetailVO();
        vo.setId(pkg.getId());
        vo.setName(pkg.getName());
        vo.setPackageType(pkg.getPackageType());
        vo.setLevelCode(pkg.getLevelCode());
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(pkg.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : pkg.getLevelCode());
        vo.setPeriod(pkg.getPeriod());
        vo.setPeriodDays(pkg.getPeriodDays());
        vo.setOriginalPrice(pkg.getOriginalPrice());
        vo.setSalePrice(pkg.getSalePrice());
        vo.setDailyPrice(pkg.getPeriodDays() != null && pkg.getPeriodDays() > 0 ? (2 * pkg.getSalePrice() + pkg.getPeriodDays()) / (2 * pkg.getPeriodDays()) : 0);
        vo.setCreditAmount(pkg.getCreditAmount());
        vo.setCreditUnitPrice(pkg.getCreditAmount() != null && pkg.getCreditAmount() > 0 ? pkg.getSalePrice() / pkg.getCreditAmount() : 0);
        vo.setDescription(pkg.getDescription());
        vo.setSalesCount(pkg.getSalesCount());
        vo.setBenefits(buildBenefits(benefit, parseBenefitOverrides(pkg.getBenefitOverrides())));
        if (withPromotions) {
            vo.setActivePromotions(getActivePromotions(pkg.getId()));
        }
        return vo;
    }

    /** 配额类权益覆盖取 max(等级权益, 覆盖值)，与履约侧口径一致 */
    private static final Set<String> BENEFIT_QUOTA_FIELDS = Set.of(
            "monthlyDehazeQuota", "monthlyDerainQuota", "monthlyDesnowQuota", "monthlyLowlightQuota",
            "monthlySuperResolutionQuota", "monthlyDenoiseQuota", "monthlyInpaintQuota", "monthlyEvaluateQuota",
            "aiCreditsDaily", "aiCreditsMonthly");

    private Map<String, Long> buildBenefits(SysMemberBenefit benefit, BenefitOverrides overrides) {
        Map<String, Long> benefits = new LinkedHashMap<>();
        if (benefit != null) {
            benefits.put("monthlyDehazeQuota", benefit.getMonthlyDehazeQuota().longValue());
            benefits.put("monthlyDerainQuota", benefit.getMonthlyDerainQuota().longValue());
            benefits.put("monthlyDesnowQuota", benefit.getMonthlyDesnowQuota().longValue());
            benefits.put("monthlyLowlightQuota", benefit.getMonthlyLowlightQuota().longValue());
            benefits.put("monthlySuperResolutionQuota", benefit.getMonthlySuperResolutionQuota().longValue());
            benefits.put("monthlyDenoiseQuota", benefit.getMonthlyDenoiseQuota().longValue());
            benefits.put("monthlyInpaintQuota", benefit.getMonthlyInpaintQuota().longValue());
            benefits.put("monthlyEvaluateQuota", benefit.getMonthlyEvaluateQuota().longValue());
            benefits.put("aiCreditsDaily", benefit.getAiCreditsDaily());
            benefits.put("aiCreditsMonthly", benefit.getAiCreditsMonthly());
            benefits.put("historyRetention", benefit.getHistoryRetention().longValue());
            benefits.put("batchLimit", benefit.getBatchLimit().longValue());
            benefits.put("priority", benefit.getPriority().longValue());
            benefits.put("advancedParams", benefit.getAdvancedParams().longValue());
            benefits.put("hdExport", benefit.getHdExport().longValue());
            benefits.put("reportExport", benefit.getReportExport().longValue());
            benefits.put("batchDownload", benefit.getBatchDownload().longValue());
        }
        if (overrides != null) {
            Map<String, Long> overridesMap = objectMapper.convertValue(overrides,
                    new TypeReference<Map<String, Long>>() {});
            overridesMap.forEach((key, value) -> {
                if (value == null) {
                    return;
                }
                Long base = benefits.get(key);
                benefits.put(key, base != null && BENEFIT_QUOTA_FIELDS.contains(key) ? Math.max(base, value) : value);
            });
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

    private List<Object> parseJsonToList(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<Object>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析JSON List失败: {}", json, e);
            return null;
        }
    }

    private Map<String, Object> parseRulesToMap(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
        } catch (JsonProcessingException e) {
            log.warn("解析JSON对象失败: {}", json, e);
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
