package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysMemberGrowthLogMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysMemberQuotaMapper;
import com.pei.dehaze.mapper.SysMemberSignInMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.entity.SysMemberGrowthLog;
import com.pei.dehaze.model.entity.SysMemberQuota;
import com.pei.dehaze.model.entity.SysMemberSignIn;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.MemberGrowthAdjustForm;
import com.pei.dehaze.model.form.MemberLevelAdjustForm;
import com.pei.dehaze.model.form.MemberStatusForm;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.model.query.GrowthLogQuery;
import com.pei.dehaze.model.query.MemberPageQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.MessageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class MemberServiceImpl extends ServiceImpl<SysMemberMapper, SysMember> implements MemberService {

    private static final int SIGN_IN_BASE_GROWTH = 3;
    private static final int SIGN_IN_BONUS_GROWTH = 20;
    private static final int SIGN_IN_BONUS_DAYS = 7;

    private static final Map<String, String> CHANGE_TYPE_LABELS = Map.of(
            "dehaze", "去雾处理",
            "evaluate", "效果评估",
            "rating", "提交评价",
            "sign_in", "每日签到",
            "sign_in_bonus", "连续签到奖励",
            "consume", "消费",
            "refund_deduct", "退款扣减",
            "admin_adjust", "管理员调整"
    );

    private final SysUserMapper userMapper;
    private final SysMemberGrowthLogMapper growthLogMapper;
    private final SysMemberSignInMapper signInMapper;
    private final SysMemberQuotaMapper quotaMapper;
    private final MemberBenefitService memberBenefitService;
    private final MessageService messageService;
    private final StringRedisTemplate stringRedisTemplate;

    @Override
    public MemberProfileVO getProfile() {
        Long userId = SecurityUtils.getUserId();
        SysMember member = getMemberOrThrow(userId);
        SysUser user = userMapper.selectById(userId);
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
        List<SysMemberBenefit> allBenefits = memberBenefitService.listAllOrdered();
        return buildProfileVO(member, user, benefit, allBenefits);
    }

    @Override
    public Page<MemberPageVO> getPage(MemberPageQuery query) {
        Page<SysMember> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysMember> wrapper = new LambdaQueryWrapper<SysMember>()
                .eq(CharSequenceUtil.isNotBlank(query.getLevelCode()), SysMember::getLevelCode, query.getLevelCode())
                .eq(query.getStatus() != null, SysMember::getStatus, query.getStatus())
                .ge(query.getExpireTimeStart() != null, SysMember::getExpireTime, query.getExpireTimeStart() != null ? query.getExpireTimeStart().atStartOfDay() : null)
                .le(query.getExpireTimeEnd() != null, SysMember::getExpireTime, query.getExpireTimeEnd() != null ? query.getExpireTimeEnd().atTime(23, 59, 59) : null)
                .ge(query.getGrowthMin() != null, SysMember::getGrowthValue, query.getGrowthMin())
                .le(query.getGrowthMax() != null, SysMember::getGrowthValue, query.getGrowthMax())
                .orderByDesc(SysMember::getBecomeMemberTime);

        if (CharSequenceUtil.isNotBlank(query.getKeywords())) {
            List<Long> userIds = userMapper.selectList(new LambdaQueryWrapper<SysUser>()
                            .and(w -> w.like(SysUser::getUsername, query.getKeywords())
                                    .or().like(SysUser::getNickname, query.getKeywords())
                                    .or().like(SysUser::getMobile, query.getKeywords())))
                    .stream()
                    .map(SysUser::getId)
                    .toList();
            if (userIds.isEmpty()) {
                Page<MemberPageVO> empty = new Page<>(page.getCurrent(), page.getSize(), 0);
                empty.setRecords(Collections.emptyList());
                return empty;
            }
            wrapper.in(SysMember::getUserId, userIds);
        }

        this.page(page, wrapper);
        List<SysMember> records = page.getRecords();
        Map<Long, SysUser> userMap;
        if (records.isEmpty()) {
            userMap = Collections.emptyMap();
        } else {
            userMap = userMapper.selectBatchIds(records.stream()
                            .map(SysMember::getUserId).distinct().toList())
                    .stream()
                    .collect(Collectors.toMap(SysUser::getId, u -> u));
        }

        Page<MemberPageVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(records.stream().map(m -> {
            SysUser u = userMap.get(m.getUserId());
            return toPageVO(m, u);
        }).toList());
        return result;
    }

    @Override
    public MemberDetailVO getDetail(Long userId) {
        SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND);
        }
        SysUser user = userMapper.selectById(userId);
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
        List<SysMemberBenefit> allBenefits = memberBenefitService.listAllOrdered();

        MemberDetailVO vo = new MemberDetailVO();
        copyProfileFields(member, user, benefit, allBenefits, vo);
        vo.setLevelSource(member.getLevelSource());
        vo.setTotalConsumption(member.getTotalConsumption());
        vo.setBecomeMemberTime(member.getBecomeMemberTime());
        vo.setFrozenReason(member.getFrozenReason());
        vo.setFrozenTime(member.getFrozenTime());
        vo.setQuotaResetMonth(member.getQuotaResetMonth());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "member", action = "level_change", targetType = "member", targetIdSpel = "#userId", beforeSpel = "", afterSpel = "#form")
    public void adjustLevel(Long userId, MemberLevelAdjustForm form) {
        SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND);
        }
        String oldLevelCode = member.getLevelCode();
        member.setLevelCode(form.getLevelCode());
        member.setLevelSource("admin");
        member.setExpireTime(form.getExpireTime());
        if (member.getBecomeMemberTime() == null) {
            member.setBecomeMemberTime(LocalDateTime.now());
        }
        SysMemberBenefit benefit = getCachedBenefit(form.getLevelCode());
        if (benefit != null) {
            member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
            member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
        }
        this.updateById(member);
        stringRedisTemplate.delete("member:level:" + userId);
        stringRedisTemplate.delete("member:quota:" + userId + ":dehaze");
        stringRedisTemplate.delete("member:quota:" + userId + ":evaluate");
        if (!form.getLevelCode().equals(oldLevelCode)) {
            sendLevelChangeNotification(userId, oldLevelCode, form.getLevelCode(), benefit);
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "member", action = "growth_change", targetType = "member", targetIdSpel = "#userId", afterSpel = "#form")
    public void adjustGrowth(Long userId, MemberGrowthAdjustForm form) {
        SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND);
        }
        Long operatorId = SecurityUtils.getUserId();
        long newGrowth = member.getGrowthValue() + form.getChangeValue();
        if (newGrowth < 0) {
            newGrowth = 0;
        }
        int actualChange = (int) (newGrowth - member.getGrowthValue());
        member.setGrowthValue(newGrowth);
        String targetLevel = calcLevelByGrowth(newGrowth);
        if (!targetLevel.equals(member.getLevelCode()) && "growth".equals(member.getLevelSource())) {
            member.setLevelCode(targetLevel);
            SysMemberBenefit benefit = memberBenefitService.getByLevelCode(targetLevel);
            if (benefit != null) {
                member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
                member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
            }
            stringRedisTemplate.delete("member:level:" + userId);
            stringRedisTemplate.delete("member:quota:" + userId + ":dehaze");
            stringRedisTemplate.delete("member:quota:" + userId + ":evaluate");
        }
        this.updateById(member);
        recordGrowthLog(userId, "admin_adjust", actualChange, newGrowth,
                null, "管理员调整成长值：" + form.getReason(), operatorId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "member", action = "status_change", targetType = "member", targetIdSpel = "#userId", afterSpel = "#form")
    public void updateStatus(Long userId, MemberStatusForm form) {
        SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND);
        }
        if (form.getStatus() == 0 && CharSequenceUtil.isBlank(form.getReason())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "冻结原因不能为空");
        }
        LambdaUpdateWrapper<SysMember> wrapper = new LambdaUpdateWrapper<SysMember>()
                .eq(SysMember::getUserId, userId)
                .set(SysMember::getStatus, form.getStatus());
        if (form.getStatus() == 0) {
            wrapper.set(SysMember::getFrozenReason, form.getReason());
            wrapper.set(SysMember::getFrozenTime, LocalDateTime.now());
        }
        this.update(wrapper);
        stringRedisTemplate.delete("member:level:" + userId);
        stringRedisTemplate.delete("member:quota:" + userId + ":dehaze");
        stringRedisTemplate.delete("member:quota:" + userId + ":evaluate");
    }

    @Override
    public Page<GrowthLogVO> getGrowthLogs(GrowthLogQuery query) {
        Long userId = SecurityUtils.getUserId();
        Page<SysMemberGrowthLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysMemberGrowthLog> wrapper = new LambdaQueryWrapper<SysMemberGrowthLog>()
                .eq(SysMemberGrowthLog::getUserId, userId)
                .eq(CharSequenceUtil.isNotBlank(query.getChangeType()), SysMemberGrowthLog::getChangeType, query.getChangeType())
                .ge(query.getStartTime() != null, SysMemberGrowthLog::getCreateTime, query.getStartTime() != null ? query.getStartTime().atStartOfDay() : null)
                .le(query.getEndTime() != null, SysMemberGrowthLog::getCreateTime, query.getEndTime() != null ? query.getEndTime().atTime(23, 59, 59) : null)
                .orderByDesc(SysMemberGrowthLog::getCreateTime);
        growthLogMapper.selectPage(page, wrapper);

        Page<GrowthLogVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toGrowthLogVO).toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SignInResultVO signIn() {
        Long userId = SecurityUtils.getUserId();
        LocalDate today = LocalDate.now();
        Long existCount = signInMapper.selectCount(new LambdaQueryWrapper<SysMemberSignIn>()
                .eq(SysMemberSignIn::getUserId, userId)
                .eq(SysMemberSignIn::getSignDate, today));
        if (existCount > 0) {
            throw new BusinessException(ResultCode.SIGN_IN_ALREADY);
        }

        SysMemberSignIn yesterdaySignIn = signInMapper.selectOne(new LambdaQueryWrapper<SysMemberSignIn>()
                .eq(SysMemberSignIn::getUserId, userId)
                .eq(SysMemberSignIn::getSignDate, today.minusDays(1))
                .last("LIMIT 1"));
        int continuousDays = yesterdaySignIn != null && yesterdaySignIn.getContinuousDays() != null
                ? yesterdaySignIn.getContinuousDays() + 1 : 1;

        boolean bonusTriggered = continuousDays > 0 && continuousDays % SIGN_IN_BONUS_DAYS == 0;
        int totalGrowth = SIGN_IN_BASE_GROWTH + (bonusTriggered ? SIGN_IN_BONUS_GROWTH : 0);

        SysMemberSignIn signIn = new SysMemberSignIn();
        signIn.setUserId(userId);
        signIn.setSignDate(today);
        signIn.setContinuousDays(continuousDays);
        signIn.setGrowthValue(totalGrowth);
        try {
            signInMapper.insert(signIn);
        } catch (DuplicateKeyException e) {
            throw new BusinessException(ResultCode.SIGN_IN_ALREADY);
        }

        SysMember member = getMemberOrThrow(userId);
        long oldGrowth = member.getGrowthValue();
        long newGrowth = oldGrowth + totalGrowth;
        member.setGrowthValue(newGrowth);
        String targetLevel = calcLevelByGrowth(newGrowth);
        if (!targetLevel.equals(member.getLevelCode()) && "growth".equals(member.getLevelSource())) {
            member.setLevelCode(targetLevel);
            SysMemberBenefit benefit = memberBenefitService.getByLevelCode(targetLevel);
            if (benefit != null) {
                member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
                member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
            }
            stringRedisTemplate.delete("member:level:" + userId);
            stringRedisTemplate.delete("member:quota:" + userId + ":dehaze");
            stringRedisTemplate.delete("member:quota:" + userId + ":evaluate");
        }
        this.updateById(member);

        recordGrowthLog(userId, "sign_in", SIGN_IN_BASE_GROWTH, oldGrowth + SIGN_IN_BASE_GROWTH,
                String.valueOf(signIn.getId()), "每日签到", null);
        if (bonusTriggered) {
            recordGrowthLog(userId, "sign_in_bonus", SIGN_IN_BONUS_GROWTH, newGrowth,
                    String.valueOf(signIn.getId()), "连续签到" + SIGN_IN_BONUS_DAYS + "天奖励", null);
        }

        SignInResultVO vo = new SignInResultVO();
        vo.setSignDate(today);
        vo.setContinuousDays(continuousDays);
        vo.setGrowthValue(SIGN_IN_BASE_GROWTH);
        vo.setBonusGrowth(bonusTriggered ? SIGN_IN_BONUS_GROWTH : 0);
        return vo;
    }

    @Override
    public SignInCalendarVO getSignInCalendar(Integer year, Integer month) {
        Long userId = SecurityUtils.getUserId();
        YearMonth ym = YearMonth.of(year, month);
        LocalDate start = ym.atDay(1);
        LocalDate end = ym.atEndOfMonth();
        List<SysMemberSignIn> records = signInMapper.selectList(new LambdaQueryWrapper<SysMemberSignIn>()
                .eq(SysMemberSignIn::getUserId, userId)
                .ge(SysMemberSignIn::getSignDate, start)
                .le(SysMemberSignIn::getSignDate, end)
                .orderByAsc(SysMemberSignIn::getSignDate));

        SignInCalendarVO vo = new SignInCalendarVO();
        vo.setSignDates(records.stream().map(SysMemberSignIn::getSignDate).toList());
        vo.setTotalDays(records.size());
        vo.setContinuousDays(records.isEmpty() ? 0 : calcCurrentContinuousDays(userId, LocalDate.now()));
        return vo;
    }

    @Override
    public void resetMonthlyQuota() {
        int currentMonth = Integer.parseInt(LocalDate.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMM")));
        Map<String, SysMemberBenefit> benefitMap = memberBenefitService.list().stream()
                .collect(Collectors.toMap(SysMemberBenefit::getLevelCode, b -> b, (a, b) -> a));

        int batchSize = 500;
        int totalProcessed = 0;
        while (true) {
            Page<SysMember> page = this.page(new Page<>(1, batchSize), new LambdaQueryWrapper<SysMember>()
                    .and(w -> w.isNull(SysMember::getQuotaResetMonth).or().ne(SysMember::getQuotaResetMonth, currentMonth)));
            List<SysMember> members = page.getRecords();
            if (members.isEmpty()) {
                break;
            }
            for (SysMember member : members) {
                if (member.getQuotaResetMonth() != null) {
                    SysMemberQuota quota = new SysMemberQuota();
                    quota.setUserId(member.getUserId());
                    quota.setQuotaMonth(member.getQuotaResetMonth());
                    quota.setLevelCode(member.getLevelCode());
                    quota.setDehazeQuota(member.getMonthlyDehazeQuota());
                    quota.setDehazeUsed(member.getMonthlyDehazeUsed());
                    quota.setEvaluateQuota(member.getMonthlyEvaluateQuota());
                    quota.setEvaluateUsed(member.getMonthlyEvaluateUsed());
                    quota.setResetTime(LocalDateTime.now());
                    quotaMapper.insert(quota);
                }
                SysMemberBenefit benefit = benefitMap.get(member.getLevelCode());
                if (benefit != null) {
                    member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
                    member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
                }
                member.setMonthlyDehazeUsed(0);
                member.setMonthlyEvaluateUsed(0);
                member.setQuotaResetMonth(currentMonth);
                this.updateById(member);
                stringRedisTemplate.delete("member:quota:" + member.getUserId() + ":dehaze");
                stringRedisTemplate.delete("member:quota:" + member.getUserId() + ":evaluate");
            }
            totalProcessed += members.size();
        }
        log.info("月度配额重置完成: 共处理{}条记录", totalProcessed);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void processExpiredMembers() {
        LocalDateTime now = LocalDateTime.now();
        List<SysMember> expired = this.list(new LambdaQueryWrapper<SysMember>()
                .lt(SysMember::getExpireTime, now)
                .ne(SysMember::getLevelSource, "growth"));
        for (SysMember member : expired) {
            String oldLevelCode = member.getLevelCode();
            String targetLevel = calcLevelByGrowth(member.getGrowthValue());
            member.setLevelCode(targetLevel);
            member.setLevelSource("growth");
            member.setExpireTime(null);
            SysMemberBenefit benefit = memberBenefitService.getByLevelCode(targetLevel);
            if (benefit != null) {
                member.setMonthlyDehazeQuota(benefit.getMonthlyDehazeQuota());
                member.setMonthlyEvaluateQuota(benefit.getMonthlyEvaluateQuota());
            }
            this.updateById(member);
            stringRedisTemplate.delete("member:level:" + member.getUserId());
            stringRedisTemplate.delete("member:quota:" + member.getUserId() + ":dehaze");
            stringRedisTemplate.delete("member:quota:" + member.getUserId() + ":evaluate");
            if (!oldLevelCode.equals(targetLevel)) {
                sendLevelChangeNotification(member.getUserId(), oldLevelCode, targetLevel, benefit);
            }
        }
        log.info("会员过期降级处理完成: 共处理{}条记录", expired.size());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void sendExpireReminders() {
        LocalDateTime now = LocalDateTime.now();
        Map<Integer, String> dayToTemplate = Map.of(
                7, "member_expire_reminder_7",
                3, "member_expire_reminder_3",
                1, "member_expire_reminder_1");
        Map<Integer, String> dayToBizPrefix = Map.of(
                7, "expire_reminder_7d",
                3, "expire_reminder_3d",
                1, "expire_reminder_1d");
        int sentCount = 0;
        for (Map.Entry<Integer, String> entry : dayToTemplate.entrySet()) {
            int days = entry.getKey();
            String templateCode = entry.getValue();
            LocalDateTime windowStart = now.plusDays(days).toLocalDate().atStartOfDay();
            LocalDateTime windowEnd = windowStart.plusDays(1);
            List<SysMember> members = this.list(new LambdaQueryWrapper<SysMember>()
                    .isNotNull(SysMember::getExpireTime)
                    .ge(SysMember::getExpireTime, windowStart)
                    .lt(SysMember::getExpireTime, windowEnd)
                    .ne(SysMember::getLevelSource, "growth"));
            if (members.isEmpty()) {
                continue;
            }
            for (SysMember member : members) {
                try {
                    MessageSendForm form = new MessageSendForm();
                    form.setType("member");
                    form.setRecipientIds(List.of(member.getUserId()));
                    form.setBizModule("member");
                    form.setBizId(dayToBizPrefix.get(days) + ":" + member.getUserId() + ":" + now.toLocalDate());
                    form.setTemplateCode(templateCode);
                    Map<String, String> variables = new HashMap<>();
                    SysMemberBenefit currentBenefit = memberBenefitService.getByLevelCode(member.getLevelCode());
                    variables.put("currentLevel", currentBenefit != null ? currentBenefit.getLevelName() : member.getLevelCode());
                    variables.put("days", String.valueOf(days));
                    variables.put("expireDate", member.getExpireTime() != null
                            ? member.getExpireTime().toLocalDate().toString() : "");
                    if (days == 3) {
                        String targetLevel = calcLevelByGrowth(member.getGrowthValue());
                        SysMemberBenefit downgradeBenefit = memberBenefitService.getByLevelCode(targetLevel);
                        variables.put("downgradeLevel", downgradeBenefit != null ? downgradeBenefit.getLevelName() : targetLevel);
                        if (currentBenefit != null && downgradeBenefit != null) {
                            variables.put("benefitCompare",
                                    "去雾:" + currentBenefit.getMonthlyDehazeQuota() + "→" + downgradeBenefit.getMonthlyDehazeQuota() + "次/月，"
                                            + "评估:" + currentBenefit.getMonthlyEvaluateQuota() + "→" + downgradeBenefit.getMonthlyEvaluateQuota() + "次/月");
                        } else {
                            variables.put("benefitCompare", "");
                        }
                    }
                    form.setVariables(variables);
                    messageService.send(form);
                    sentCount++;
                } catch (Exception e) {
                    log.warn("到期提醒发送失败: userId={}, days={}", member.getUserId(), days, e);
                }
            }
        }
        log.info("会员到期预警完成: 共发送{}条提醒", sentCount);
    }

    @Override
    @AuditLog(module = "member", action = "quota_deduct", targetType = "member", targetIdSpel = "#userId", afterSpel = "{quotaType:#quotaType,amount:#amount}")
    public boolean deductQuota(Long userId, String quotaType, int amount) {
        if (amount <= 0) {
            return true;
        }
        String quotaKey = "member:quota:" + userId + ":" + quotaType;
        String cached = stringRedisTemplate.opsForValue().get(quotaKey);
        if (cached == null) {
            SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                    .eq(SysMember::getUserId, userId));
            if (member == null) {
                return false;
            }
            if (member.getStatus() != 1) {
                throw new BusinessException(ResultCode.MEMBER_FROZEN);
            }
            int remaining = "dehaze".equals(quotaType)
                    ? (member.getMonthlyDehazeQuota() != null ? member.getMonthlyDehazeQuota() : 0)
                    - (member.getMonthlyDehazeUsed() != null ? member.getMonthlyDehazeUsed() : 0)
                    : (member.getMonthlyEvaluateQuota() != null ? member.getMonthlyEvaluateQuota() : 0)
                    - (member.getMonthlyEvaluateUsed() != null ? member.getMonthlyEvaluateUsed() : 0);
            stringRedisTemplate.opsForValue().set(quotaKey, String.valueOf(remaining), Duration.ofHours(25));
            cached = String.valueOf(remaining);
        }
        Long remaining = Long.parseLong(cached);
        if (remaining < amount) {
            return false;
        }
        Long newRemaining = stringRedisTemplate.opsForValue().decrement(quotaKey, amount);
        if (newRemaining == null || newRemaining < 0) {
            stringRedisTemplate.opsForValue().increment(quotaKey, amount);
            return false;
        }
        updateQuotaUsedInDb(userId, quotaType, amount);
        return true;
    }

    private void updateQuotaUsedInDb(Long userId, String quotaType, int amount) {
        try {
            SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                    .eq(SysMember::getUserId, userId));
            if (member == null) {
                return;
            }
            if ("dehaze".equals(quotaType)) {
                member.setMonthlyDehazeUsed((member.getMonthlyDehazeUsed() != null ? member.getMonthlyDehazeUsed() : 0) + amount);
            } else {
                member.setMonthlyEvaluateUsed((member.getMonthlyEvaluateUsed() != null ? member.getMonthlyEvaluateUsed() : 0) + amount);
            }
            this.updateById(member);
        } catch (Exception e) {
            log.error("配额落库失败，等待补偿: userId={}, quotaType={}", userId, quotaType, e);
        }
    }

    private SysMemberBenefit getCachedBenefit(String levelCode) {
        String cacheKey = "member:benefit:" + levelCode;
        String cached = stringRedisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            return cn.hutool.json.JSONUtil.toBean(cached, SysMemberBenefit.class);
        }
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(levelCode);
        if (benefit != null) {
            stringRedisTemplate.opsForValue().set(cacheKey, cn.hutool.json.JSONUtil.toJsonStr(benefit), Duration.ofMinutes(60));
        }
        return benefit;
    }

    private void sendLevelChangeNotification(Long userId, String oldLevelCode, String newLevelCode, SysMemberBenefit newBenefit) {
        try {
            SysMemberBenefit oldBenefit = memberBenefitService.getByLevelCode(oldLevelCode);
            int oldSort = oldBenefit != null && oldBenefit.getSort() != null ? oldBenefit.getSort() : 0;
            int newSort = newBenefit != null && newBenefit.getSort() != null ? newBenefit.getSort() : 0;
            MessageSendForm form = new MessageSendForm();
            form.setType("member");
            form.setRecipientIds(List.of(userId));
            form.setBizModule("member");
            form.setBizId("level_change:" + userId + ":" + System.currentTimeMillis());
            if (newSort > oldSort) {
                form.setTemplateCode("member_level_up");
                Map<String, String> variables = new HashMap<>();
                variables.put("levelName", newBenefit != null ? newBenefit.getLevelName() : newLevelCode);
                variables.put("benefitList", newBenefit != null ? "去雾" + newBenefit.getMonthlyDehazeQuota() + "次/月，评估" + newBenefit.getMonthlyEvaluateQuota() + "次/月" : "");
                form.setVariables(variables);
            } else {
                form.setTemplateCode("member_downgrade_warning");
                Map<String, String> variables = new HashMap<>();
                variables.put("currentLevel", oldBenefit != null ? oldBenefit.getLevelName() : oldLevelCode);
                variables.put("days", "0");
                variables.put("downgradeLevel", newBenefit != null ? newBenefit.getLevelName() : newLevelCode);
                form.setVariables(variables);
            }
            messageService.send(form);
        } catch (Exception e) {
            log.warn("等级变更通知发送失败: userId={}, old={}, new={}", userId, oldLevelCode, newLevelCode, e);
        }
    }

    private SysMember getMemberOrThrow(Long userId) {
        SysMember member = this.getOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(ResultCode.MEMBER_NOT_FOUND);
        }
        return member;
    }

    private String calcLevelByGrowth(long growth) {
        List<SysMemberBenefit> benefits = memberBenefitService.listAllOrdered();
        return benefits.stream()
                .filter(b -> growth >= b.getGrowthMin())
                .filter(b -> b.getGrowthMax() == 0 || growth <= b.getGrowthMax())
                .max(Comparator.comparingLong(SysMemberBenefit::getGrowthMin))
                .map(SysMemberBenefit::getLevelCode)
                .orElse("level_0");
    }

    private int calcCurrentContinuousDays(Long userId, LocalDate today) {
        Long todayCount = signInMapper.selectCount(new LambdaQueryWrapper<SysMemberSignIn>()
                .eq(SysMemberSignIn::getUserId, userId)
                .eq(SysMemberSignIn::getSignDate, today));
        LocalDate checkDate = todayCount > 0 ? today : today.minusDays(1);
        int days = 0;
        while (true) {
            Long count = signInMapper.selectCount(new LambdaQueryWrapper<SysMemberSignIn>()
                    .eq(SysMemberSignIn::getUserId, userId)
                    .eq(SysMemberSignIn::getSignDate, checkDate));
            if (count == 0) {
                break;
            }
            days++;
            checkDate = checkDate.minusDays(1);
        }
        return days;
    }

    private void recordGrowthLog(Long userId, String changeType, int changeValue, long balance,
                                 String relatedId, String reason, Long operatorId) {
        SysMemberGrowthLog log = new SysMemberGrowthLog();
        log.setUserId(userId);
        log.setChangeType(changeType);
        log.setChangeValue(changeValue);
        log.setBalance(balance);
        log.setRelatedId(relatedId);
        log.setReason(reason);
        log.setOperatorId(operatorId);
        growthLogMapper.insert(log);
    }

    private MemberProfileVO buildProfileVO(SysMember member, SysUser user, SysMemberBenefit benefit,
                                           List<SysMemberBenefit> allBenefits) {
        MemberProfileVO vo = new MemberProfileVO();
        copyProfileFields(member, user, benefit, allBenefits, vo);
        return vo;
    }

    private void copyProfileFields(SysMember member, SysUser user, SysMemberBenefit benefit,
                                   List<SysMemberBenefit> allBenefits, MemberProfileVO vo) {
        vo.setUserId(member.getUserId());
        if (user != null) {
            vo.setUsername(user.getUsername());
            vo.setNickname(user.getNickname());
            vo.setAvatar(user.getAvatar());
        }
        vo.setLevelCode(member.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : member.getLevelCode());
        vo.setGrowthValue(member.getGrowthValue());
        vo.setExpireTime(member.getExpireTime());
        vo.setMonthlyDehazeQuota(member.getMonthlyDehazeQuota());
        vo.setMonthlyDehazeUsed(member.getMonthlyDehazeUsed());
        vo.setMonthlyEvaluateQuota(member.getMonthlyEvaluateQuota());
        vo.setMonthlyEvaluateUsed(member.getMonthlyEvaluateUsed());
        vo.setStatus(member.getStatus());
        vo.setBenefits(toBenefitVO(benefit));
        fillProgress(member, benefit, allBenefits, vo);
    }

    private void fillProgress(SysMember member, SysMemberBenefit current, List<SysMemberBenefit> allBenefits, MemberProfileVO vo) {
        if (current == null || allBenefits == null || allBenefits.isEmpty()) {
            vo.setProgressPercent(0);
            return;
        }
        List<SysMemberBenefit> ordered = allBenefits.stream()
                .sorted(Comparator.comparingLong(SysMemberBenefit::getGrowthMin))
                .toList();
        int currentIdx = -1;
        for (int i = 0; i < ordered.size(); i++) {
            if (ordered.get(i).getLevelCode().equals(member.getLevelCode())) {
                currentIdx = i;
                break;
            }
        }
        if (currentIdx < 0 || currentIdx >= ordered.size() - 1) {
            vo.setProgressPercent(100);
            return;
        }
        SysMemberBenefit next = ordered.get(currentIdx + 1);
        long currentMin = current.getGrowthMin();
        long nextMin = next.getGrowthMin();
        long growth = member.getGrowthValue();
        if (growth >= nextMin) {
            vo.setProgressPercent(100);
            vo.setNextLevelGrowth(0L);
            return;
        }
        long range = nextMin - currentMin;
        long progress = growth - currentMin;
        int percent = range > 0 ? (int) (progress * 100 / range) : 0;
        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;
        vo.setProgressPercent(percent);
        vo.setNextLevelGrowth(nextMin - growth);
    }

    private BenefitVO toBenefitVO(SysMemberBenefit benefit) {
        if (benefit == null) {
            return null;
        }
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

    private MemberPageVO toPageVO(SysMember member, SysUser user) {
        MemberPageVO vo = new MemberPageVO();
        vo.setUserId(member.getUserId());
        if (user != null) {
            vo.setUsername(user.getUsername());
            vo.setNickname(user.getNickname());
        }
        vo.setLevelCode(member.getLevelCode());
        SysMemberBenefit benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
        vo.setLevelName(benefit != null ? benefit.getLevelName() : member.getLevelCode());
        vo.setGrowthValue(member.getGrowthValue());
        vo.setMonthlyUsed((member.getMonthlyDehazeUsed() != null ? member.getMonthlyDehazeUsed() : 0)
                + (member.getMonthlyEvaluateUsed() != null ? member.getMonthlyEvaluateUsed() : 0));
        vo.setExpireTime(member.getExpireTime());
        vo.setStatus(member.getStatus());
        vo.setBecomeMemberTime(member.getBecomeMemberTime());
        return vo;
    }

    private GrowthLogVO toGrowthLogVO(SysMemberGrowthLog log) {
        GrowthLogVO vo = new GrowthLogVO();
        vo.setId(log.getId());
        vo.setChangeType(log.getChangeType());
        vo.setChangeTypeLabel(CHANGE_TYPE_LABELS.getOrDefault(log.getChangeType(), log.getChangeType()));
        vo.setChangeValue(log.getChangeValue());
        vo.setBalance(log.getBalance());
        vo.setRelatedId(log.getRelatedId());
        vo.setReason(log.getReason());
        vo.setOperatorId(log.getOperatorId());
        vo.setCreateTime(log.getCreateTime());
        return vo;
    }
}
