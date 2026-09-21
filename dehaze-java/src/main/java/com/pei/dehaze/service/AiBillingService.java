package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiDateTimeUtils;
import com.pei.dehaze.common.util.AiJsonUtils;
import com.pei.dehaze.mapper.AiBillingInsightMapper;
import com.pei.dehaze.mapper.AiBillingLedgerMapper;
import com.pei.dehaze.mapper.SysAiBillingAnomalyMapper;
import com.pei.dehaze.mapper.SysAiBillingMapper;
import com.pei.dehaze.mapper.SysAiCreditLogMapper;
import com.pei.dehaze.mapper.SysAiModelCostDetailMapper;
import com.pei.dehaze.mapper.SysAiModelCostMapper;
import com.pei.dehaze.mapper.SysAiRefundMapper;
import com.pei.dehaze.mapper.SysMemberBenefitMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.entity.SysAiBilling;
import com.pei.dehaze.model.entity.SysAiBillingAnomaly;
import com.pei.dehaze.model.entity.SysAiCreditLog;
import com.pei.dehaze.model.entity.SysAiModelCost;
import com.pei.dehaze.model.entity.SysAiModelCostDetail;
import com.pei.dehaze.model.entity.SysAiRefund;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.form.AiBillingAdjustForm;
import com.pei.dehaze.model.form.AiModelCostForm;
import com.pei.dehaze.model.form.AiModelCostUpdateForm;
import com.pei.dehaze.model.form.AiReconcileImportForm;
import com.pei.dehaze.model.form.AiRefundAuditForm;
import com.pei.dehaze.model.form.AiRefundCreateForm;
import com.pei.dehaze.model.query.AiBillingAnomalyQuery;
import com.pei.dehaze.model.query.AiBillingRecordQuery;
import com.pei.dehaze.model.query.AiBillingStatQuery;
import com.pei.dehaze.model.query.AiCreditLogQuery;
import com.pei.dehaze.model.query.AiModelCostQuery;
import com.pei.dehaze.model.query.AiRefundQuery;
import com.pei.dehaze.model.read.AiBillTypeRead;
import com.pei.dehaze.model.read.AiBillingStatRead;
import com.pei.dehaze.model.read.AiBillingModelRead;
import com.pei.dehaze.model.read.AiBillingPeriodRead;
import com.pei.dehaze.model.read.AiCostStatRead;
import com.pei.dehaze.model.read.AiCreditSourceRead;
import com.pei.dehaze.model.read.AiOrderIncomeRead;
import com.pei.dehaze.model.read.AiUserCreditsRead;
import com.pei.dehaze.model.vo.AiBalanceVO;
import com.pei.dehaze.model.vo.AiBillVO;
import com.pei.dehaze.model.vo.AiBillingAnomalyVO;
import com.pei.dehaze.model.vo.AiBillingRecordVO;
import com.pei.dehaze.model.vo.AiBillingStatVO;
import com.pei.dehaze.model.vo.AiBillingSummaryVO;
import com.pei.dehaze.model.vo.AiCostStatVO;
import com.pei.dehaze.model.vo.AiCreditLogVO;
import com.pei.dehaze.model.vo.AiModelCostVO;
import com.pei.dehaze.model.vo.AiRefundVO;
import com.pei.dehaze.security.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * AI 计费管理服务（用户端余额/明细/账单/退款 + 管理端统计/调整/审核/成本）。
 *
 * <p>对齐 dehaze-python {@code app/service/billing/*}：余额与配额以 Redis 为准实时权威
 * （{@code ai:balance:*} / {@code ai:arrears:*} / {@code ai:quota:*}），MySQL 为持久化权威
 * （sys_user.credits_balance 乐观锁 CAS），账单缓存 {@code ai:bill:*} 以 snake_case JSON 与
 * python 互认。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiBillingService {

    private static final ZoneId BILLING_ZONE = ZoneId.of("Asia/Shanghai");

    private static final DateTimeFormatter DAY_SUFFIX = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private static final DateTimeFormatter MONTH_SUFFIX = DateTimeFormatter.ofPattern("yyyy-MM");

    /** 月份解析：python 用 strptime("%Y-%m")，月可 1 位（2026-1），此处单列解析格式与之对齐 */
    private static final DateTimeFormatter MONTH_PARSE = DateTimeFormatter.ofPattern("yyyy-M");

    private static final String BALANCE_KEY = "ai:balance:%d";

    private static final String ARREARS_KEY = "ai:arrears:%d";

    private static final String QUOTA_DAILY_KEY = "ai:quota:daily:%d:%s";

    private static final String QUOTA_MONTHLY_KEY = "ai:quota:monthly:%d:%s";

    private static final String BILL_KEY = "ai:bill:%d:%s";

    private static final Duration BALANCE_TTL = Duration.ofHours(1);

    /** 账单缓存 TTL：python AI_BILLING_BILL_CACHE_TTL = 90 天 */
    private static final Duration BILL_CACHE_TTL = Duration.ofDays(90);

    private static final int CAS_RETRY = 3;

    /** 计入充值的流水来源（vip_gift_expire 为过期为负值，一并计入充减） */
    private static final Set<String> RECHARGE_SOURCES =
            Set.of("recharge", "vip_gift", "trial", "admin_adjust", "vip_gift_expire");

    /** chat 类计费类型：token 口径统计仅基于 chat 类（asr/tts 的 input_tokens 存秒数/字符数） */
    private static final List<String> CHAT_BILL_TYPES = List.of("chat", "chat_subagent");

    private static final String STAT_PERMISSION = "ai:billing:stat";

    private static final int MODEL_DIST_TOP = 5;

    private final SysAiBillingMapper billingMapper;

    private final SysAiCreditLogMapper creditLogMapper;

    private final SysAiRefundMapper refundMapper;

    private final SysAiBillingAnomalyMapper anomalyMapper;

    private final SysAiModelCostMapper modelCostMapper;

    private final SysAiModelCostDetailMapper modelCostDetailMapper;

    private final AiBillingInsightMapper insightMapper;

    private final AiBillingLedgerMapper ledgerMapper;

    private final SysUserMapper userMapper;

    private final SysMemberMapper memberMapper;

    private final SysMemberBenefitMapper memberBenefitMapper;

    private final StringRedisTemplate stringRedisTemplate;

    // ── 用户端 ──────────────────────────────────────────────

    public AiBalanceVO getBalance(Long userIdParam) {
        return buildBalance(resolveQueryUser(userIdParam));
    }

    public AiBillingSummaryVO getSummary(String dimension) {
        Long userId = SecurityUtils.getUserId();
        LocalDateTime now = LocalDateTime.now();
        // create_time 以秒级精度入库（MySQL DATETIME 对微秒四舍五入），上界取下一整秒
        LocalDateTime nowCeiling = now.plusSeconds(1).withNano(0);
        LocalDateTime periodStart;
        String fmt;
        if ("month".equals(dimension)) {
            periodStart = now.withDayOfMonth(1).toLocalDate().atStartOfDay();
            fmt = "%Y-%m";
        } else if ("day".equals(dimension)) {
            periodStart = now.toLocalDate().atStartOfDay();
            fmt = "%Y-%m-%d";
        } else {
            throw new BusinessException(ResultCode.PARAM_ERROR, "dimension 仅支持 day/month");
        }

        List<AiBillingPeriodRead> rows =
                insightMapper.sumGroupByPeriod(userId, periodStart, nowCeiling, fmt);
        List<AiBillingSummaryVO.TrendPoint> trend = new ArrayList<>();
        int inputTokens = 0;
        int outputTokens = 0;
        int cachedInputTokens = 0;
        int creditsSaved = 0;
        int totalCredits = 0;
        for (AiBillingPeriodRead row : rows) {
            AiBillingSummaryVO.TrendPoint point = new AiBillingSummaryVO.TrendPoint();
            point.setDate(row.getDate());
            point.setCredits(intOf(row.getCredits()));
            point.setInputTokens(intOf(row.getInputTokens()));
            point.setOutputTokens(intOf(row.getOutputTokens()));
            trend.add(point);
            totalCredits += point.getCredits();
            inputTokens += point.getInputTokens();
            outputTokens += point.getOutputTokens();
            cachedInputTokens += intOf(row.getCachedInputTokens());
            creditsSaved += intOf(row.getCreditsSaved());
        }

        List<AiBillingModelRead> distRows = new ArrayList<>(insightMapper.sumGroupByModel(userId, periodStart, nowCeiling));
        distRows.sort(Comparator.comparingLong((AiBillingModelRead r) -> longOf(r.getCredits())).reversed());
        List<AiBillingSummaryVO.ModelDist> distribution = new ArrayList<>();
        for (AiBillingModelRead row : distRows.subList(0, Math.min(MODEL_DIST_TOP, distRows.size()))) {
            AiBillingSummaryVO.ModelDist item = new AiBillingSummaryVO.ModelDist();
            item.setModel(row.getModel());
            item.setCredits(intOf(row.getCredits()));
            item.setTokens(intOf(row.getInputTokens()) + intOf(row.getOutputTokens()));
            distribution.add(item);
        }

        AiBillingSummaryVO.Savings savings = new AiBillingSummaryVO.Savings();
        savings.setCachedInputTokens(cachedInputTokens);
        savings.setCreditsSaved(creditsSaved);

        AiBillingSummaryVO vo = new AiBillingSummaryVO();
        vo.setTotalCredits(totalCredits);
        vo.setInputTokens(inputTokens);
        vo.setOutputTokens(outputTokens);
        vo.setTrend(trend);
        vo.setModelDistribution(distribution);
        vo.setSavings(savings);
        return vo;
    }

    public IPage<AiBillingRecordVO> listRecords(AiBillingRecordQuery query) {
        Long userId = resolveQueryUser(query.getUserId());
        LambdaQueryWrapper<SysAiBilling> wrapper = new LambdaQueryWrapper<SysAiBilling>()
                .eq(SysAiBilling::getUserId, userId)
                .eq(query.getConversationId() != null, SysAiBilling::getConversationId, query.getConversationId())
                .eq(org.springframework.util.StringUtils.hasText(query.getBillType()),
                        SysAiBilling::getBillType, query.getBillType())
                .eq(org.springframework.util.StringUtils.hasText(query.getModelId()),
                        SysAiBilling::getModel, query.getModelId())
                .ge(query.getDateStart() != null && !query.getDateStart().isBlank(),
                        SysAiBilling::getCreateTime, AiDateTimeUtils.parse(query.getDateStart()))
                .le(query.getDateEnd() != null && !query.getDateEnd().isBlank(),
                        SysAiBilling::getCreateTime, AiDateTimeUtils.parse(query.getDateEnd()))
                .orderByDesc(SysAiBilling::getCreateTime)
                .orderByDesc(SysAiBilling::getId);
        Page<SysAiBilling> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiBilling> result = billingMapper.selectPage(page, wrapper);
        List<Long> billingIds = result.getRecords().stream().map(SysAiBilling::getId).toList();
        Map<Long, Integer> refundStatus = latestRefundStatus(billingIds);
        List<AiBillingRecordVO> records = new ArrayList<>();
        for (SysAiBilling billing : result.getRecords()) {
            AiBillingRecordVO vo = toRecordVO(billing);
            vo.setRefundStatus(refundStatus.getOrDefault(billing.getId(), 0));
            records.add(vo);
        }
        return pageOf(page, records, result.getTotal());
    }

    public IPage<AiCreditLogVO> listCreditLogs(AiCreditLogQuery query) {
        Long userId = resolveQueryUser(query.getUserId());
        LambdaQueryWrapper<SysAiCreditLog> wrapper = new LambdaQueryWrapper<SysAiCreditLog>()
                .eq(SysAiCreditLog::getUserId, userId)
                .eq(hasText(query.getSource()), SysAiCreditLog::getSource, query.getSource())
                .ge(hasText(query.getDateStart()), SysAiCreditLog::getCreateTime,
                        AiDateTimeUtils.parse(query.getDateStart()))
                .le(hasText(query.getDateEnd()), SysAiCreditLog::getCreateTime,
                        AiDateTimeUtils.parse(query.getDateEnd()))
                .orderByDesc(SysAiCreditLog::getCreateTime)
                .orderByDesc(SysAiCreditLog::getId);
        Page<SysAiCreditLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiCreditLog> result = creditLogMapper.selectPage(page, wrapper);
        List<AiCreditLogVO> records = new ArrayList<>();
        for (SysAiCreditLog log : result.getRecords()) {
            AiCreditLogVO vo = new AiCreditLogVO();
            vo.setId(log.getId());
            vo.setUserId(log.getUserId());
            vo.setSource(log.getSource());
            vo.setAmount(log.getAmount());
            vo.setBalanceAfter(log.getBalanceAfter());
            vo.setRelatedId(log.getRelatedId());
            vo.setReason(log.getReason());
            vo.setOperatorId(log.getOperatorId());
            vo.setCreateTime(log.getCreateTime());
            records.add(vo);
        }
        return pageOf(page, records, result.getTotal());
    }

    public AiBillVO getBill(String month) {
        Long userId = SecurityUtils.getUserId();
        AiBillVO cached = readBillCache(userId, month);
        if (cached != null) {
            if (isEmptyBill(cached, month)) {
                // 历史空账期缓存已失效（如缓存写入后又被清空）：清掉，避免后续误判为存在
                stringRedisTemplate.delete(BILL_KEY.formatted(userId, month));
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "账单不存在");
            }
            return cached;
        }
        AiBillVO bill = computeBill(userId, month);
        // 空账期不入缓存：否则后续查询命中全 0 缓存会错误地返回成功而非 A0401
        if (isEmptyBill(bill, month)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "账单不存在");
        }
        stringRedisTemplate.opsForValue()
                .set(BILL_KEY.formatted(userId, month), AiJsonUtils.write(bill), BILL_CACHE_TTL);
        return bill;
    }

    @Transactional
    public AiRefundVO applyRefund(AiRefundCreateForm form) {
        Long userId = SecurityUtils.getUserId();
        if (form.getAmount() == null || form.getAmount() <= 0) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "退款积分数必须大于 0");
        }
        SysAiBilling record = billingMapper.selectById(form.getBillingId());
        if (record == null || !Objects.equals(record.getUserId(), userId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "计费记录不存在");
        }
        if (form.getAmount() > (record.getCredits() == null ? 0 : record.getCredits())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "退款积分数超过该笔记录实际消耗");
        }
        if (pendingRefund(form.getBillingId()) != null) {
            throw new BusinessException(ResultCode.AI_REFUND_ALREADY_EXISTS);
        }
        SysAiRefund refund = new SysAiRefund();
        refund.setUserId(userId);
        refund.setBillingId(form.getBillingId());
        refund.setAmount(form.getAmount());
        refund.setReason(form.getReason());
        refund.setStatus(1);
        refund.setCreateBy(userId);
        refundMapper.insert(refund);
        return toRefundVO(refund);
    }

    // ── 管理端 ──────────────────────────────────────────────

    public IPage<AiRefundVO> listRefunds(AiRefundQuery query) {
        LambdaQueryWrapper<SysAiRefund> wrapper = new LambdaQueryWrapper<SysAiRefund>()
                .eq(query.getStatus() != null, SysAiRefund::getStatus, query.getStatus())
                .eq(query.getUserId() != null, SysAiRefund::getUserId, query.getUserId())
                .ge(hasText(query.getDateStart()), SysAiRefund::getCreateTime,
                        AiDateTimeUtils.parse(query.getDateStart()))
                .le(hasText(query.getDateEnd()), SysAiRefund::getCreateTime,
                        AiDateTimeUtils.parse(query.getDateEnd()))
                .orderByDesc(SysAiRefund::getCreateTime)
                .orderByDesc(SysAiRefund::getId);
        Page<SysAiRefund> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiRefund> result = refundMapper.selectPage(page, wrapper);
        return pageOf(page, result.getRecords().stream().map(this::toRefundVO).toList(), result.getTotal());
    }

    public List<AiBillingStatVO> getStats(AiBillingStatQuery query) {
        List<String> dimensions = List.of("user", "model", "billType", "day");
        if (!dimensions.contains(query.getGroupBy())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "groupBy 仅支持 user/model/billType/day");
        }
        List<AiBillingStatRead> rows = insightMapper.statsByDimension(
                query.getGroupBy(),
                query.getUserId(),
                query.getModelId(),
                query.getBillType(),
                AiDateTimeUtils.parse(query.getDateStart()),
                AiDateTimeUtils.parse(query.getDateEnd()));
        List<AiBillingStatVO> results = new ArrayList<>();
        for (AiBillingStatRead row : rows) {
            long chatInput = longOf(row.getTotalInputTokens());
            AiBillingStatVO vo = new AiBillingStatVO();
            vo.setDimension(row.getDimension());
            vo.setTotalCredits(intOf(row.getTotalCredits()));
            vo.setTotalInputTokens(intOf(row.getTotalInputTokens()));
            vo.setTotalOutputTokens(intOf(row.getTotalOutputTokens()));
            vo.setCacheHitRate(chatInput > 0
                    ? BigDecimal.valueOf(longOf(row.getChatCachedTokens()))
                            .divide(BigDecimal.valueOf(chatInput), 4, RoundingMode.HALF_UP).doubleValue()
                    : 0.0);
            vo.setCreditsSaved(intOf(row.getCreditsSaved()));
            vo.setDegradationCount(intOf(row.getDegradationCount()));
            results.add(vo);
        }
        return results;
    }

    @Transactional
    public AiBalanceVO adjustCredits(AiBillingAdjustForm form) {
        BigDecimal amount = form.getAmount();
        if (amount == null || amount.signum() == 0) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "调整积分数不能为 0");
        }
        // 积分是整数口径（python AdjustRequest.amount: int），小数必须拒绝而非截断
        if (amount.stripTrailingZeros().scale() > 0) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "调整积分数必须为整数");
        }
        if (userMapper.selectById(form.getUserId()) == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在");
        }
        increaseBalance(form.getUserId(), amount.intValueExact(), "admin_adjust", null, form.getReason(),
                SecurityUtils.getUserId());
        return buildBalance(form.getUserId());
    }

    @Transactional
    public AiRefundVO auditRefund(Long refundId, AiRefundAuditForm form) {
        Long operatorId = SecurityUtils.getUserId();
        SysAiRefund refund = refundMapper.selectById(refundId);
        if (refund == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "退款申请不存在");
        }
        if (!Integer.valueOf(1).equals(refund.getStatus())) {
            throw new BusinessException(ResultCode.REFUND_AUDIT_FAILED, "该退款申请已审核");
        }
        if (Boolean.TRUE.equals(form.getApproved())) {
            // A0681 校验：原计费记录必须存在且未补偿过（防重复回补余额）
            if (billingMapper.selectById(refund.getBillingId()) == null) {
                throw new BusinessException(ResultCode.REFUND_AUDIT_FAILED, "原计费记录不存在");
            }
            Long approvedExists = refundMapper.selectCount(new LambdaQueryWrapper<SysAiRefund>()
                    .eq(SysAiRefund::getBillingId, refund.getBillingId())
                    .eq(SysAiRefund::getStatus, 2)
                    .ne(SysAiRefund::getId, refund.getId()));
            if (approvedExists != null && approvedExists > 0) {
                throw new BusinessException(ResultCode.REFUND_AUDIT_FAILED, "原计费记录已退款");
            }
            // 余额回补（Redis INCR + MySQL CAS + 流水），不调整配额已用计数
            increaseBalance(refund.getUserId(), refund.getAmount(), "refund", refund.getBillingId(),
                    "退款: " + refund.getReason(), operatorId);
            refund.setStatus(2);
        } else {
            refund.setStatus(3);
        }
        refund.setAuditorId(operatorId);
        refund.setAuditRemark(form.getAuditRemark());
        refundMapper.updateById(refund);
        return toRefundVO(refundMapper.selectById(refund.getId()));
    }

    public IPage<AiBillingAnomalyVO> listAnomalies(AiBillingAnomalyQuery query) {
        LocalDateTime start = AiDateTimeUtils.parse(query.getDateStart());
        LocalDateTime end = AiDateTimeUtils.parse(query.getDateEnd());
        LambdaQueryWrapper<SysAiBillingAnomaly> wrapper = new LambdaQueryWrapper<SysAiBillingAnomaly>()
                .eq(query.getUserId() != null, SysAiBillingAnomaly::getUserId, query.getUserId())
                .eq(hasText(query.getAnomalyType()), SysAiBillingAnomaly::getAnomalyType, query.getAnomalyType())
                .eq(query.getStatus() != null, SysAiBillingAnomaly::getStatus, query.getStatus())
                .ge(start != null, SysAiBillingAnomaly::getTriggerAt, start)
                .lt(end != null, SysAiBillingAnomaly::getTriggerAt, end)
                .orderByDesc(SysAiBillingAnomaly::getTriggerAt)
                .orderByDesc(SysAiBillingAnomaly::getId);
        Page<SysAiBillingAnomaly> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiBillingAnomaly> result = anomalyMapper.selectPage(page, wrapper);
        List<AiBillingAnomalyVO> records = new ArrayList<>();
        for (SysAiBillingAnomaly anomaly : result.getRecords()) {
            AiBillingAnomalyVO vo = new AiBillingAnomalyVO();
            vo.setId(anomaly.getId());
            vo.setUserId(anomaly.getUserId());
            vo.setBillingId(anomaly.getBillingId());
            vo.setAnomalyType(anomaly.getAnomalyType());
            vo.setDetail(anomaly.getDetail());
            vo.setStatus(anomaly.getStatus());
            vo.setTriggerAt(anomaly.getTriggerAt());
            vo.setCreateTime(anomaly.getCreateTime());
            records.add(vo);
        }
        return pageOf(page, records, result.getTotal());
    }

    public IPage<AiModelCostVO> listCosts(AiModelCostQuery query) {
        LambdaQueryWrapper<SysAiModelCost> wrapper = new LambdaQueryWrapper<SysAiModelCost>()
                .eq(hasText(query.getModelId()), SysAiModelCost::getModelId, query.getModelId())
                .eq(query.getProviderId() != null, SysAiModelCost::getProviderId, query.getProviderId())
                .like(hasText(query.getKeyword()), SysAiModelCost::getModelId, query.getKeyword())
                .orderByDesc(SysAiModelCost::getCreateTime)
                .orderByDesc(SysAiModelCost::getId);
        Page<SysAiModelCost> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiModelCost> result = modelCostMapper.selectPage(page, wrapper);
        List<AiModelCostVO> records = new ArrayList<>();
        for (SysAiModelCost cost : result.getRecords()) {
            records.add(toCostVO(cost, listCostDetails(cost.getId())));
        }
        return pageOf(page, records, result.getTotal());
    }

    @Transactional
    public AiModelCostVO createCost(AiModelCostForm form) {
        SysAiModelCost cost = new SysAiModelCost();
        cost.setModelId(form.getModelId());
        cost.setProviderId(form.getProviderId());
        cost.setPriceVersion(modelCostMapper.nextPriceVersion(form.getModelId(), form.getProviderId()));
        cost.setCurrency(hasText(form.getCurrency()) ? form.getCurrency() : "CNY");
        cost.setEffectiveFrom(form.getEffectiveFrom() != null
                ? form.getEffectiveFrom() : LocalDateTime.now(BILLING_ZONE).withNano(0));
        cost.setEffectiveTo(form.getEffectiveTo());
        cost.setStatus(form.getStatus() == null ? 1 : form.getStatus());
        modelCostMapper.insert(cost);

        List<SysAiModelCostDetail> details = new ArrayList<>();
        for (AiModelCostForm.Detail detail : form.getDetails()) {
            SysAiModelCostDetail entity = new SysAiModelCostDetail();
            entity.setPriceId(cost.getId());
            entity.setTokenType(detail.getTokenType());
            entity.setTimeSlot(detail.getTimeSlot());
            entity.setMinTokens(detail.getMinTokens() == null ? 0L : detail.getMinTokens());
            entity.setMaxTokens(detail.getMaxTokens());
            entity.setUnitPrice(detail.getUnitPrice());
            modelCostDetailMapper.insert(entity);
            details.add(entity);
        }
        return toCostVO(cost, details);
    }

    @Transactional
    public AiModelCostVO updateCost(Long costId, AiModelCostUpdateForm form) {
        SysAiModelCost cost = modelCostMapper.selectById(costId);
        if (cost == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "成本单价不存在");
        }
        if (hasText(form.getCurrency())) {
            cost.setCurrency(form.getCurrency());
        }
        if (form.getEffectiveFrom() != null) {
            cost.setEffectiveFrom(form.getEffectiveFrom());
        }
        if (form.getEffectiveTo() != null) {
            cost.setEffectiveTo(form.getEffectiveTo());
        }
        if (form.getStatus() != null) {
            cost.setStatus(form.getStatus());
        }
        modelCostMapper.updateById(cost);
        return toCostVO(cost, listCostDetails(costId));
    }

    @Transactional
    public void deleteCost(Long costId) {
        if (modelCostMapper.selectById(costId) == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "成本单价不存在");
        }
        modelCostMapper.deleteById(costId);
        modelCostDetailMapper.delete(new LambdaQueryWrapper<SysAiModelCostDetail>()
                .eq(SysAiModelCostDetail::getPriceId, costId));
    }

    public List<AiCostStatVO> getCostStats(String startTime, String endTime, String groupBy,
                                           String modelId, Long providerId) {
        LocalDateTime start = AiDateTimeUtils.parse(startTime);
        LocalDateTime end = AiDateTimeUtils.parse(endTime);
        if ("overall".equals(groupBy)) {
            return overallCostStats(start, end);
        }
        if ("model".equals(groupBy) || "provider".equals(groupBy)) {
            List<AiCostStatVO> results = new ArrayList<>();
            for (AiCostStatRead row : insightMapper.sumCostGroupBy(groupBy, start, end, modelId, providerId)) {
                AiCostStatVO vo = new AiCostStatVO();
                vo.setDimension(row.getDimension());
                vo.setCost(scale2(row.getCost()).doubleValue());
                results.add(vo);
            }
            return results;
        }
        throw new BusinessException(ResultCode.PARAM_ERROR, "groupBy 仅支持 overall/model/provider");
    }

    public Map<String, Object> importReconcile(AiReconcileImportForm form) {
        int imported = (int) form.getContent().lines().filter(line -> !line.isBlank()).count();
        return Map.of("imported", imported);
    }

    // ── 内部实现 ────────────────────────────────────────────

    /**
     * 解析查询目标用户：管理员可指定 userId 查询他人（需 ai:billing:stat），普通用户仅可查本人
     */
    private Long resolveQueryUser(Long userIdParam) {
        Long current = SecurityUtils.getUserId();
        if (userIdParam == null || Objects.equals(userIdParam, current)) {
            return current;
        }
        if (SecurityUtils.isRoot() || SecurityUtils.getPerms().contains(STAT_PERMISSION)) {
            return userIdParam;
        }
        throw new BusinessException(ResultCode.ACCESS_UNAUTHORIZED, "访问未授权");
    }

    private AiBalanceVO buildBalance(Long userId) {
        int[] used = getQuotaUsed(userId);
        long[] limits = getQuotaLimits(userId);
        AiBalanceVO vo = new AiBalanceVO();
        vo.setUserId(userId);
        vo.setCreditsBalance(readBalance(userId));
        vo.setArrearsStatus(stringRedisTemplate.hasKey(ARREARS_KEY.formatted(userId)));
        vo.setDailyUsed(used[0]);
        vo.setDailyLimit((int) (limits == null ? 0 : limits[0]));
        vo.setMonthlyUsed(used[1]);
        vo.setMonthlyLimit((int) (limits == null ? 0 : limits[1]));
        return vo;
    }

    /**
     * 查询余额：Redis 优先，未命中查 MySQL 并整数化回填。
     *
     * <p>缓存值必须是整数字面量：python 侧 DECRBY/INCRBY 对 "100.00" 这类历史坏值会报错，
     * 故与 python {@code int(val)} 同口径判定，坏值删除后回源整数化回填。
     */
    private BigDecimal readBalance(Long userId) {
        String key = BALANCE_KEY.formatted(userId);
        String cached = stringRedisTemplate.opsForValue().get(key);
        if (cached != null && cached.matches("-?\\d+")) {
            return new BigDecimal(cached);
        }
        if (cached != null) {
            stringRedisTemplate.delete(key);
        }
        AiUserCreditsRead current = ledgerMapper.getCreditsBalanceAndVersion(userId);
        BigDecimal balance = current == null || current.getCreditsBalance() == null
                ? BigDecimal.ZERO : current.getCreditsBalance();
        stringRedisTemplate.opsForValue().set(key, balance.toBigInteger().toString(), BALANCE_TTL);
        return balance;
    }

    private int[] getQuotaUsed(Long userId) {
        LocalDate today = LocalDate.now(BILLING_ZONE);
        String daily = stringRedisTemplate.opsForValue()
                .get(QUOTA_DAILY_KEY.formatted(userId, today.format(DAY_SUFFIX)));
        String monthly = stringRedisTemplate.opsForValue()
                .get(QUOTA_MONTHLY_KEY.formatted(userId, today.format(MONTH_SUFFIX)));
        return new int[]{intOf(daily), intOf(monthly)};
    }

    /**
     * 日/月限额取用户会员等级的启用权益；无会员或权益缺失/停用返回 null（fail-closed，展示为 0）
     */
    private long[] getQuotaLimits(Long userId) {
        SysMember member = memberMapper.selectOne(new LambdaQueryWrapper<SysMember>()
                .eq(SysMember::getUserId, userId).last("LIMIT 1"));
        if (member == null) {
            return null;
        }
        SysMemberBenefit benefit = memberBenefitMapper.selectOne(new LambdaQueryWrapper<SysMemberBenefit>()
                .eq(SysMemberBenefit::getLevelCode, member.getLevelCode()).last("LIMIT 1"));
        if (benefit == null || !Integer.valueOf(1).equals(benefit.getStatus())) {
            return null;
        }
        return new long[]{longOf(benefit.getAiCreditsDaily()), longOf(benefit.getAiCreditsMonthly())};
    }

    /**
     * 增加余额：Redis INCRBY → MySQL CAS 落库 → 清欠费标记 → 写流水
     */
    private void increaseBalance(Long userId, Integer amount, String source, Long relatedId,
                                 String reason, Long operatorId) {
        stringRedisTemplate.opsForValue().increment(BALANCE_KEY.formatted(userId), amount);
        increaseBalanceCas(userId, BigDecimal.valueOf(amount));
        stringRedisTemplate.delete(ARREARS_KEY.formatted(userId));
        BigDecimal balance = readBalance(userId);
        SysAiCreditLog log = new SysAiCreditLog();
        log.setUserId(userId);
        log.setSource(source);
        log.setAmount((long) amount);
        log.setBalanceAfter(balance.toBigInteger().longValue());
        log.setRelatedId(relatedId);
        log.setReason(reason);
        log.setOperatorId(operatorId);
        creditLogMapper.insert(log);
    }

    private void increaseBalanceCas(Long userId, BigDecimal amount) {
        for (int i = 0; i < CAS_RETRY; i++) {
            AiUserCreditsRead current = ledgerMapper.getCreditsBalanceAndVersion(userId);
            if (current == null) {
                // Redis 已加而 MySQL 无账户：静默跳过会造成余额永久背离，必须显式失败
                throw new BusinessException(ResultCode.BUSINESS_ERROR,
                        "余额增加落库失败（用户余额账户不存在）: user_id=" + userId);
            }
            if (ledgerMapper.increaseBalanceCas(userId, amount, current.getCreditsVersion()) > 0) {
                return;
            }
        }
        throw new BusinessException(ResultCode.BUSINESS_ERROR,
                "余额增加落库失败（CAS 重试耗尽）: user_id=" + userId);
    }

    private Map<Long, Integer> latestRefundStatus(List<Long> billingIds) {
        if (billingIds.isEmpty()) {
            return Map.of();
        }
        List<SysAiRefund> refunds = refundMapper.selectList(new LambdaQueryWrapper<SysAiRefund>()
                .select(SysAiRefund::getBillingId, SysAiRefund::getStatus)
                .in(SysAiRefund::getBillingId, billingIds)
                .orderByAsc(SysAiRefund::getId));
        Map<Long, Integer> statusMap = new HashMap<>();
        for (SysAiRefund refund : refunds) {
            statusMap.put(refund.getBillingId(), refund.getStatus());
        }
        return statusMap;
    }

    private SysAiRefund pendingRefund(Long billingId) {
        return refundMapper.selectOne(new LambdaQueryWrapper<SysAiRefund>()
                .eq(SysAiRefund::getBillingId, billingId)
                .eq(SysAiRefund::getStatus, 1)
                .last("LIMIT 1"));
    }

    private AiBillVO readBillCache(Long userId, String month) {
        parseMonth(month);
        String key = BILL_KEY.formatted(userId, month);
        String cached = stringRedisTemplate.opsForValue().get(key);
        if (cached == null) {
            return null;
        }
        AiBillVO bill = AiJsonUtils.read(cached, AiBillVO.class);
        if (bill == null) {
            // 缓存坏值（非账单 JSON）：删除后重算，避免 500
            stringRedisTemplate.delete(key);
        }
        return bill;
    }

    private AiBillVO computeBill(Long userId, String month) {
        LocalDateTime[] bounds = monthBounds(month);
        Map<String, Integer> itemSummary = new LinkedHashMap<>();
        int totalConsume = 0;
        for (AiBillTypeRead row : insightMapper.sumByBillType(userId, bounds[0], bounds[1])) {
            int credits = intOf(row.getCredits());
            itemSummary.put(row.getBillType(), credits);
            totalConsume += credits;
        }

        Map<String, BigDecimal> bySource = new HashMap<>();
        for (AiCreditSourceRead row : insightMapper.sumCreditLogBySource(userId, bounds[0], bounds[1])) {
            bySource.put(row.getSource(), row.getAmount() == null ? BigDecimal.ZERO : row.getAmount());
        }
        BigDecimal rechargeAmount = bySource.entrySet().stream()
                .filter(entry -> RECHARGE_SOURCES.contains(entry.getKey()))
                .map(Map.Entry::getValue)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        int totalRecharge = rechargeAmount.toBigInteger().intValue();
        int totalRefund = bySource.getOrDefault("refund", BigDecimal.ZERO).toBigInteger().intValue();

        AiBillVO bill = new AiBillVO();
        bill.setUserId(userId);
        bill.setMonth(month);
        bill.setTotalConsume(totalConsume);
        bill.setTotalRecharge(totalRecharge);
        bill.setTotalRefund(totalRefund);
        bill.setBalanceStart(balanceAtOrBefore(userId, bounds[0].minusSeconds(1)));
        bill.setBalanceEnd(balanceAtOrBefore(userId, bounds[1]));
        bill.setItemSummary(itemSummary);
        return bill;
    }

    private BigDecimal balanceAtOrBefore(Long userId, LocalDateTime time) {
        BigDecimal balance = insightMapper.getBalanceAtOrBefore(userId, time);
        return balance == null ? BigDecimal.ZERO : balance;
    }

    /** 非当前月份且无任何消费/充值/退款记录视为账单不存在（当前月份允许全 0） */
    private boolean isEmptyBill(AiBillVO bill, String month) {
        if (month.equals(YearMonth.now(BILLING_ZONE).format(MONTH_SUFFIX))) {
            return false;
        }
        return bill.getTotalConsume() == 0 && bill.getTotalRecharge() == 0 && bill.getTotalRefund() == 0;
    }

    private LocalDateTime[] monthBounds(String month) {
        LocalDateTime start = parseMonth(month);
        return new LocalDateTime[]{start, start.plusMonths(1).minusSeconds(1)};
    }

    private LocalDateTime parseMonth(String month) {
        try {
            return YearMonth.parse(month, MONTH_PARSE).atDay(1).atStartOfDay();
        } catch (DateTimeParseException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "月份格式不正确，应为 YYYY-MM");
        }
    }

    private List<SysAiModelCostDetail> listCostDetails(Long priceId) {
        return modelCostDetailMapper.selectList(new LambdaQueryWrapper<SysAiModelCostDetail>()
                .eq(SysAiModelCostDetail::getPriceId, priceId)
                .orderByAsc(SysAiModelCostDetail::getId));
    }

    private List<AiCostStatVO> overallCostStats(LocalDateTime start, LocalDateTime end) {
        BigDecimal totalCost = insightMapper.sumCost(start, end);

        BigDecimal creditIncome = BigDecimal.ZERO;
        BigDecimal vipIncome = BigDecimal.ZERO;
        for (AiOrderIncomeRead row : insightMapper.sumPaidOrderByPackageType(start, end)) {
            // 订单金额单位为分，转元
            BigDecimal revenue = BigDecimal.valueOf(longOf(row.getAmount()))
                    .divide(BigDecimal.valueOf(100), 2, RoundingMode.HALF_UP);
            if ("credit".equals(row.getPackageType())) {
                creditIncome = creditIncome.add(revenue);
            } else {
                vipIncome = vipIncome.add(revenue);
            }
        }
        // 会员卡实收按 AI 分摊比例 30% 计入 AI 参考口径（其余为其他业务线收入）
        BigDecimal aiIncome = creditIncome.add(vipIncome.multiply(new BigDecimal("0.3")));

        List<AiCostStatVO> results = new ArrayList<>();
        results.add(buildCostStat("overall", creditIncome.add(vipIncome), totalCost));
        results.add(buildCostStat("ai", aiIncome, totalCost));
        return results;
    }

    private AiCostStatVO buildCostStat(String metric, BigDecimal revenue, BigDecimal cost) {
        BigDecimal profit = revenue.subtract(cost);
        AiCostStatVO vo = new AiCostStatVO();
        vo.setMetric(metric);
        vo.setRevenue(scale2(revenue).doubleValue());
        vo.setCost(scale2(cost).doubleValue());
        vo.setProfit(scale2(profit).doubleValue());
        vo.setProfitRate(revenue.signum() == 0 ? 0.0
                : profit.divide(revenue, 4, RoundingMode.HALF_UP).doubleValue());
        return vo;
    }

    private AiBillingRecordVO toRecordVO(SysAiBilling billing) {
        AiBillingRecordVO vo = new AiBillingRecordVO();
        vo.setId(billing.getId());
        vo.setUserId(billing.getUserId());
        vo.setConversationId(billing.getConversationId());
        vo.setMessageId(billing.getMessageId());
        vo.setModel(billing.getModel());
        vo.setActualModel(billing.getActualModel());
        vo.setBillType(billing.getBillType());
        vo.setInputTokens(billing.getInputTokens());
        vo.setCachedInputTokens(billing.getCachedInputTokens());
        vo.setOutputTokens(billing.getOutputTokens());
        vo.setCredits(billing.getCredits());
        vo.setCreditsSaved(billing.getCreditsSaved());
        vo.setToolCredits(billing.getToolCredits());
        vo.setQuotaConsumed(billing.getQuotaConsumed());
        vo.setPreDeduct(billing.getPreDeduct());
        vo.setCreateTime(billing.getCreateTime());
        return vo;
    }

    private AiRefundVO toRefundVO(SysAiRefund refund) {
        AiRefundVO vo = new AiRefundVO();
        vo.setId(refund.getId());
        vo.setUserId(refund.getUserId());
        vo.setBillingId(refund.getBillingId());
        vo.setAmount(refund.getAmount());
        vo.setReason(refund.getReason());
        vo.setStatus(refund.getStatus());
        vo.setAuditorId(refund.getAuditorId());
        vo.setAuditRemark(refund.getAuditRemark());
        vo.setCreateTime(refund.getCreateTime());
        vo.setUpdateTime(refund.getUpdateTime());
        return vo;
    }

    private AiModelCostVO toCostVO(SysAiModelCost cost, List<SysAiModelCostDetail> details) {
        AiModelCostVO vo = new AiModelCostVO();
        vo.setId(cost.getId());
        vo.setModelId(cost.getModelId());
        vo.setProviderId(cost.getProviderId());
        vo.setPriceVersion(cost.getPriceVersion());
        vo.setCurrency(cost.getCurrency());
        vo.setEffectiveFrom(cost.getEffectiveFrom());
        vo.setEffectiveTo(cost.getEffectiveTo());
        vo.setStatus(cost.getStatus());
        vo.setCreateTime(cost.getCreateTime());
        vo.setUpdateTime(cost.getUpdateTime());
        List<AiModelCostVO.Detail> items = new ArrayList<>();
        for (SysAiModelCostDetail detail : details) {
            AiModelCostVO.Detail item = new AiModelCostVO.Detail();
            item.setId(detail.getId());
            item.setPriceId(detail.getPriceId());
            item.setTokenType(detail.getTokenType());
            item.setTimeSlot(detail.getTimeSlot());
            item.setMinTokens(detail.getMinTokens());
            item.setMaxTokens(detail.getMaxTokens());
            item.setUnitPrice(detail.getUnitPrice());
            items.add(item);
        }
        vo.setDetails(items);
        return vo;
    }

    private static boolean hasText(String value) {
        return org.springframework.util.StringUtils.hasText(value);
    }

    private static BigDecimal scale2(BigDecimal value) {
        return (value == null ? BigDecimal.ZERO : value).setScale(2, RoundingMode.HALF_UP);
    }

    private static int intOf(Object value) {
        return (int) Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, longOf(value)));
    }

    private static long longOf(Object value) {
        if (value == null) {
            return 0L;
        }
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return Long.parseLong(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private static <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(new ArrayList<>(records));
        return result;
    }
}
