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
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysUserCouponMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysCoupon;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserCoupon;
import com.pei.dehaze.model.form.CouponBatchDistributeForm;
import com.pei.dehaze.model.form.CouponForm;
import com.pei.dehaze.model.query.CouponPageQuery;
import com.pei.dehaze.model.vo.CouponBatchResult;
import com.pei.dehaze.model.vo.CouponCreateResult;
import com.pei.dehaze.model.vo.CouponReceiveResult;
import com.pei.dehaze.model.vo.CouponVO;
import com.pei.dehaze.model.vo.UserCouponVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.CouponService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class CouponServiceImpl extends ServiceImpl<SysCouponMapper, SysCoupon> implements CouponService {

    private final SysUserCouponMapper userCouponMapper;
    private final SysUserMapper userMapper;
    private final SysMemberMapper memberMapper;
    private final ObjectMapper objectMapper;
    private final StringRedisTemplate stringRedisTemplate;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public CouponCreateResult create(CouponForm form) {
        validateCouponForm(form);
        SysCoupon coupon = new SysCoupon();
        coupon.setName(form.getName());
        coupon.setType(form.getType());
        coupon.setFaceValue(form.getFaceValue());
        coupon.setThreshold(form.getThreshold());
        coupon.setValidType(form.getValidType());
        coupon.setValidStart(form.getValidStart());
        coupon.setValidEnd(form.getValidEnd());
        coupon.setValidDays(form.getValidDays());
        coupon.setTotalQty(form.getTotalQty());
        coupon.setIssuedQty(0);
        coupon.setUsedQty(0);
        coupon.setPerUserLimit(form.getPerUserLimit());
        coupon.setApplicableScope(serializeListToJson(form.getApplicableScope()));
        coupon.setStatus(form.getStatus() != null ? form.getStatus() : 1);
        this.save(coupon);
        return new CouponCreateResult(coupon.getId());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Long id, CouponForm form) {
        SysCoupon coupon = this.getById(id);
        if (coupon == null) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        validateCouponForm(form);
        coupon.setName(form.getName());
        coupon.setType(form.getType());
        coupon.setFaceValue(form.getFaceValue());
        coupon.setThreshold(form.getThreshold());
        coupon.setValidType(form.getValidType());
        coupon.setValidStart(form.getValidStart());
        coupon.setValidEnd(form.getValidEnd());
        coupon.setValidDays(form.getValidDays());
        coupon.setTotalQty(form.getTotalQty());
        coupon.setPerUserLimit(form.getPerUserLimit());
        coupon.setApplicableScope(serializeListToJson(form.getApplicableScope()));
        if (form.getStatus() != null) {
            coupon.setStatus(form.getStatus());
        }
        this.updateById(coupon);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteByIds(String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).filter(s -> !s.isEmpty()).map(Long::parseLong).toList();
        if (idList.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "未指定删除的优惠券ID");
        }
        for (Long id : idList) {
            SysCoupon coupon = this.getById(id);
            if (coupon == null) {
                throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
            }
            Long usedCount = userCouponMapper.selectCount(new LambdaQueryWrapper<SysUserCoupon>()
                    .eq(SysUserCoupon::getCouponId, id)
                    .eq(SysUserCoupon::getStatus, 2));
            if (usedCount > 0) {
                throw new BusinessException(ResultCode.DATA_BIND_EXISTS, "优惠券已发放使用，无法删除");
            }
            userCouponMapper.delete(new LambdaQueryWrapper<SysUserCoupon>()
                    .eq(SysUserCoupon::getCouponId, id)
                    .eq(SysUserCoupon::getStatus, 1));
            this.removeById(id);
        }
    }

    @Override
    public CouponBatchResult batchDistribute(CouponBatchDistributeForm form) {
        SysCoupon coupon = this.getById(form.getCouponId());
        if (coupon == null) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        if (coupon.getStatus() != 1) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND, "优惠券已禁用");
        }
        List<Long> targetUserIds = resolveTargetUserIds(form.getTargetScope(), form.getLevelCodes(), form.getUserIds());
        int successCount = 0;
        int failCount = 0;
        for (Long userId : targetUserIds) {
            try {
                distributeToUser(coupon, userId);
                successCount++;
            } catch (Exception e) {
                log.warn("批量发放优惠券失败: couponId={}, userId={}, err={}", coupon.getId(), userId, e.getMessage());
                failCount++;
            }
        }
        return new CouponBatchResult(successCount, failCount);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public CouponReceiveResult receive(Long couponId) {
        Long userId = SecurityUtils.getUserId();
        SysCoupon coupon = this.getById(couponId);
        if (coupon == null || coupon.getStatus() != 1) {
            throw new BusinessException(ResultCode.COUPON_NOT_FOUND);
        }
        LocalDateTime now = LocalDateTime.now();
        if ("fixed".equals(coupon.getValidType()) && coupon.getValidEnd() != null && now.isAfter(coupon.getValidEnd())) {
            throw new BusinessException(ResultCode.COUPON_EXPIRED);
        }
        String rateLimitKey = "coupon:receive:rate:" + userId;
        Long count = stringRedisTemplate.opsForValue().increment(rateLimitKey);
        if (count != null && count == 1) {
            stringRedisTemplate.expire(rateLimitKey, 60, TimeUnit.SECONDS);
        }
        if (count != null && count > 5) {
            throw new BusinessException(ResultCode.RATE_LIMIT);
        }
        Long receivedCount = userCouponMapper.selectCount(new LambdaQueryWrapper<SysUserCoupon>()
                .eq(SysUserCoupon::getUserId, userId)
                .eq(SysUserCoupon::getCouponId, couponId));
        if (receivedCount >= coupon.getPerUserLimit()) {
            throw new BusinessException(ResultCode.COUPON_LIMIT_EXCEEDED);
        }

        int rows = baseMapper.update(null, new LambdaUpdateWrapper<SysCoupon>()
                .eq(SysCoupon::getId, couponId)
                .eq(SysCoupon::getStatus, 1)
                .and(w -> w.eq(SysCoupon::getTotalQty, -1).or().apply("issued_qty < total_qty"))
                .setSql("issued_qty = issued_qty + 1"));
        if (rows == 0) {
            throw new BusinessException(ResultCode.COUPON_STOCK_EMPTY);
        }

        SysUserCoupon userCoupon = new SysUserCoupon();
        userCoupon.setUserId(userId);
        userCoupon.setCouponId(couponId);
        userCoupon.setStatus(1);
        userCoupon.setReceiveTime(now);
        if ("relative".equals(coupon.getValidType()) && coupon.getValidDays() != null) {
            userCoupon.setExpireTime(now.plusDays(coupon.getValidDays()));
        } else if ("fixed".equals(coupon.getValidType())) {
            userCoupon.setExpireTime(coupon.getValidEnd());
        }
        userCouponMapper.insert(userCoupon);

        return new CouponReceiveResult(userCoupon.getId());
    }

    @Override
    public List<UserCouponVO> listMy(Integer status) {
        Long userId = SecurityUtils.getUserId();
        LambdaQueryWrapper<SysUserCoupon> wrapper = new LambdaQueryWrapper<SysUserCoupon>()
                .eq(SysUserCoupon::getUserId, userId)
                .eq(status != null, SysUserCoupon::getStatus, status)
                .orderByDesc(SysUserCoupon::getReceiveTime);
        List<SysUserCoupon> userCoupons = userCouponMapper.selectList(wrapper);
        if (userCoupons.isEmpty()) {
            return Collections.emptyList();
        }
        Map<Long, SysCoupon> couponMap = baseMapper.selectByIdsIncludeDeleted(
                        userCoupons.stream().map(SysUserCoupon::getCouponId).distinct().toList())
                .stream()
                .collect(Collectors.toMap(SysCoupon::getId, c -> c));
        return userCoupons.stream().map(uc -> toUserCouponVO(uc, couponMap.get(uc.getCouponId()))).toList();
    }

    @Override
    public Page<CouponVO> getPage(CouponPageQuery query) {
        Page<SysCoupon> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysCoupon> wrapper = new LambdaQueryWrapper<SysCoupon>()
                .like(CharSequenceUtil.isNotBlank(query.getName()), SysCoupon::getName, query.getName())
                .eq(CharSequenceUtil.isNotBlank(query.getType()), SysCoupon::getType, query.getType())
                .eq(query.getStatus() != null, SysCoupon::getStatus, query.getStatus())
                .orderByDesc(SysCoupon::getId);
        this.page(page, wrapper);

        Page<CouponVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toCouponVO).toList());
        return result;
    }

    private void distributeToUser(SysCoupon coupon, Long userId) {
        LocalDateTime now = LocalDateTime.now();
        Long receivedCount = userCouponMapper.selectCount(new LambdaQueryWrapper<SysUserCoupon>()
                .eq(SysUserCoupon::getUserId, userId)
                .eq(SysUserCoupon::getCouponId, coupon.getId()));
        if (receivedCount >= coupon.getPerUserLimit()) {
            throw new BusinessException(ResultCode.COUPON_LIMIT_EXCEEDED);
        }

        int rows = baseMapper.update(null, new LambdaUpdateWrapper<SysCoupon>()
                .eq(SysCoupon::getId, coupon.getId())
                .eq(SysCoupon::getStatus, 1)
                .and(w -> w.eq(SysCoupon::getTotalQty, -1).or().apply("issued_qty < total_qty"))
                .setSql("issued_qty = issued_qty + 1"));
        if (rows == 0) {
            throw new BusinessException(ResultCode.COUPON_STOCK_EMPTY);
        }

        SysUserCoupon userCoupon = new SysUserCoupon();
        userCoupon.setUserId(userId);
        userCoupon.setCouponId(coupon.getId());
        userCoupon.setStatus(1);
        userCoupon.setReceiveTime(now);
        if ("relative".equals(coupon.getValidType()) && coupon.getValidDays() != null) {
            userCoupon.setExpireTime(now.plusDays(coupon.getValidDays()));
        } else if ("fixed".equals(coupon.getValidType())) {
            userCoupon.setExpireTime(coupon.getValidEnd());
        }
        userCouponMapper.insert(userCoupon);
    }

    private List<Long> resolveTargetUserIds(String targetScope, List<String> levelCodes, List<Long> userIds) {
        if ("all".equals(targetScope)) {
            return userMapper.selectList(new LambdaQueryWrapper<SysUser>().eq(SysUser::getStatus, 1))
                    .stream().map(SysUser::getId).toList();
        }
        if ("level".equals(targetScope) && levelCodes != null && !levelCodes.isEmpty()) {
            List<Long> memberUserIds = memberMapper.selectList(new LambdaQueryWrapper<SysMember>()
                            .in(SysMember::getLevelCode, levelCodes))
                    .stream().map(SysMember::getUserId).distinct().toList();
            if (memberUserIds.isEmpty()) {
                return Collections.emptyList();
            }
            return userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .in(SysUser::getId, memberUserIds)
                            .eq(SysUser::getStatus, 1))
                    .stream().map(SysUser::getId).toList();
        }
        if ("users".equals(targetScope) && userIds != null && !userIds.isEmpty()) {
            return userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .in(SysUser::getId, userIds)
                            .eq(SysUser::getStatus, 1))
                    .stream().map(SysUser::getId).toList();
        }
        return Collections.emptyList();
    }

    private void validateCouponForm(CouponForm form) {
        if ("full_reduction".equals(form.getType()) && (form.getThreshold() == null || form.getThreshold() < 0)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "满减券必须设置使用门槛");
        }
        if ("fixed".equals(form.getValidType()) && (form.getValidStart() == null || form.getValidEnd() == null)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "固定有效期必须设置起止时间");
        }
        if ("relative".equals(form.getValidType()) && (form.getValidDays() == null || form.getValidDays() < 1)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "相对有效期必须设置有效天数");
        }
    }

    private CouponVO toCouponVO(SysCoupon coupon) {
        CouponVO vo = new CouponVO();
        vo.setId(coupon.getId());
        vo.setName(coupon.getName());
        vo.setType(coupon.getType());
        vo.setFaceValue(coupon.getFaceValue());
        vo.setThreshold(coupon.getThreshold());
        vo.setValidType(coupon.getValidType());
        vo.setValidStart(coupon.getValidStart());
        vo.setValidEnd(coupon.getValidEnd());
        vo.setValidDays(coupon.getValidDays());
        vo.setTotalQty(coupon.getTotalQty());
        vo.setIssuedQty(coupon.getIssuedQty());
        vo.setUsedQty(coupon.getUsedQty());
        vo.setPerUserLimit(coupon.getPerUserLimit());
        vo.setApplicableScope(parseJsonToList(coupon.getApplicableScope()));
        vo.setStatus(coupon.getStatus());
        vo.setCreateTime(coupon.getCreateTime());
        return vo;
    }

    private UserCouponVO toUserCouponVO(SysUserCoupon userCoupon, SysCoupon coupon) {
        UserCouponVO vo = new UserCouponVO();
        vo.setId(userCoupon.getId());
        vo.setCouponId(userCoupon.getCouponId());
        vo.setCouponName(coupon != null ? coupon.getName() : null);
        vo.setType(coupon != null ? coupon.getType() : null);
        vo.setFaceValue(coupon != null ? coupon.getFaceValue() : null);
        vo.setThreshold(coupon != null ? coupon.getThreshold() : null);
        vo.setStatus(userCoupon.getStatus());
        vo.setReceiveTime(userCoupon.getReceiveTime());
        vo.setExpireTime(userCoupon.getExpireTime());
        vo.setUsedTime(userCoupon.getUsedTime());
        vo.setUsedOrderId(userCoupon.getUsedOrderId());
        vo.setApplicableScope(coupon != null ? parseJsonToList(coupon.getApplicableScope()) : null);
        return vo;
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

    private String serializeListToJson(List<Long> list) {
        if (list == null || list.isEmpty()) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(list);
        } catch (JsonProcessingException e) {
            log.warn("序列化List到JSON失败", e);
            return null;
        }
    }
}
