package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
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
import com.pei.dehaze.model.entity.SysAiCreditLog;
import com.pei.dehaze.model.entity.SysAiModelCost;
import com.pei.dehaze.model.entity.SysAiModelCostDetail;
import com.pei.dehaze.model.entity.SysAiRefund;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.AiBillingAdjustForm;
import com.pei.dehaze.model.form.AiModelCostForm;
import com.pei.dehaze.model.form.AiReconcileImportForm;
import com.pei.dehaze.model.form.AiRefundAuditForm;
import com.pei.dehaze.model.form.AiRefundCreateForm;
import com.pei.dehaze.model.query.AiBillingStatQuery;
import com.pei.dehaze.model.read.AiBillingStatRead;
import com.pei.dehaze.model.read.AiCostStatRead;
import com.pei.dehaze.model.read.AiOrderIncomeRead;
import com.pei.dehaze.model.read.AiUserCreditsRead;
import com.pei.dehaze.model.vo.AiBalanceVO;
import com.pei.dehaze.model.vo.AiBillVO;
import com.pei.dehaze.model.vo.AiBillingStatVO;
import com.pei.dehaze.model.vo.AiCostStatVO;
import com.pei.dehaze.model.vo.AiModelCostVO;
import com.pei.dehaze.model.vo.AiRefundVO;
import com.pei.dehaze.security.util.SecurityUtils;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.math.BigDecimal;
import java.time.Duration;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AiBillingService 单元测试。
 *
 * <p>锁定与 dehaze-python 一致的关键口径：账单缓存 snake_case 互认与空账期 A0401、
 * 退款校验与审核回补、余额 Redis+MySQL CAS 双写、成本双口径毛利与统计维度校验。
 */
@DisplayName("AiBillingService 单元测试")
@ExtendWith(MockitoExtension.class)
class AiBillingServiceTest {

    private static final ZoneId BILLING_ZONE = ZoneId.of("Asia/Shanghai");

    @Mock
    private SysAiBillingMapper billingMapper;

    @Mock
    private SysAiCreditLogMapper creditLogMapper;

    @Mock
    private SysAiRefundMapper refundMapper;

    @Mock
    private SysAiBillingAnomalyMapper anomalyMapper;

    @Mock
    private SysAiModelCostMapper modelCostMapper;

    @Mock
    private SysAiModelCostDetailMapper modelCostDetailMapper;

    @Mock
    private AiBillingInsightMapper insightMapper;

    @Mock
    private AiBillingLedgerMapper ledgerMapper;

    @Mock
    private SysUserMapper userMapper;

    @Mock
    private SysMemberMapper memberMapper;

    @Mock
    private SysMemberBenefitMapper memberBenefitMapper;

    @Mock
    private StringRedisTemplate stringRedisTemplate;

    @Mock
    private ValueOperations<String, String> valueOperations;

    @InjectMocks
    private AiBillingService service;

    /** 无 Spring 上下文时 LambdaQueryWrapper 的 select(...) 需要先注册实体表信息 */
    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiRefund.class);
    }

    // ── 账单 ────────────────────────────────────────────────

    @Test
    @DisplayName("账单缓存命中：按 python snake_case JSON 解析并直接返回")
    void getBill_returnsCachedSnakeCasePayload() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("ai:bill:1:2026-07")).thenReturn(
                "{\"user_id\":1,\"month\":\"2026-07\",\"total_consume\":100,\"total_recharge\":0,"
                        + "\"total_refund\":0,\"balance_start\":0,\"balance_end\":0,"
                        + "\"item_summary\":{\"chat\":100}}");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            AiBillVO bill = service.getBill("2026-07");

            assertThat(bill.getTotalConsume()).isEqualTo(100);
            assertThat(bill.getItemSummary()).containsEntry("chat", 100);
        }
        verify(insightMapper, never()).sumByBillType(anyLong(), any(), any());
    }

    @Test
    @DisplayName("历史空账期抛 A0401 且不写缓存（后续查询不会命中全 0 缓存）")
    void getBill_emptyHistoryMonthNotCached() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("ai:bill:1:2020-01")).thenReturn(null);
        when(insightMapper.sumByBillType(eq(1L), any(), any())).thenReturn(List.of());
        when(insightMapper.sumCreditLogBySource(eq(1L), any(), any())).thenReturn(List.of());
        when(insightMapper.getBalanceAtOrBefore(eq(1L), any())).thenReturn(null);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.getBill("2020-01"))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
        verify(valueOperations, never()).set(eq("ai:bill:1:2020-01"), anyString(), any(Duration.class));
    }

    @Test
    @DisplayName("陈旧空账期缓存命中：抛 A0401 并清除该缓存键")
    void getBill_staleEmptyCacheEvicted() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("ai:bill:1:2020-01")).thenReturn(
                "{\"user_id\":1,\"month\":\"2020-01\",\"total_consume\":0,\"total_recharge\":0,"
                        + "\"total_refund\":0,\"balance_start\":0,\"balance_end\":0,\"item_summary\":{}}");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.getBill("2020-01"))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
        verify(stringRedisTemplate).delete("ai:bill:1:2020-01");
    }

    @Test
    @DisplayName("当前月份允许全 0（月初无数据属正常），不抛账单不存在")
    void getBill_currentMonthAllowsZeros() {
        String month = YearMonth.now(BILLING_ZONE).format(DateTimeFormatter.ofPattern("yyyy-MM"));
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("ai:bill:1:" + month)).thenReturn(null);
        when(insightMapper.sumByBillType(eq(1L), any(), any())).thenReturn(List.of());
        when(insightMapper.sumCreditLogBySource(eq(1L), any(), any())).thenReturn(List.of());
        when(insightMapper.getBalanceAtOrBefore(eq(1L), any())).thenReturn(BigDecimal.ZERO);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            AiBillVO bill = service.getBill(month);

            assertThat(bill.getTotalConsume()).isZero();
            assertThat(bill.getMonth()).isEqualTo(month);
        }
        verify(valueOperations).set(eq("ai:bill:1:" + month), anyString(), any(Duration.class));
    }

    @Test
    @DisplayName("月份格式非法抛 A0400")
    void getBill_rejectsInvalidMonth() {
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.getBill("2026/07"))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
        }
    }

    // ── 退款 ────────────────────────────────────────────────

    @Test
    @DisplayName("退款申请：积分数必须大于 0")
    void applyRefund_rejectsNonPositiveAmount() {
        AiRefundCreateForm form = new AiRefundCreateForm();
        form.setBillingId(3L);
        form.setAmount(0);
        form.setReason("误扣");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.applyRefund(form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
        }
        verify(billingMapper, never()).selectById(any());
    }

    @Test
    @DisplayName("退款申请：非本人计费记录按不存在处理（A0401，不暴露他人记录存在性）")
    void applyRefund_rejectsForeignBilling() {
        SysAiBilling billing = new SysAiBilling();
        billing.setId(3L);
        billing.setUserId(99L);
        when(billingMapper.selectById(3L)).thenReturn(billing);
        AiRefundCreateForm form = new AiRefundCreateForm();
        form.setBillingId(3L);
        form.setAmount(10);
        form.setReason("误扣");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.applyRefund(form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
    }

    @Test
    @DisplayName("退款申请：退款积分超过该笔记录实际消耗抛 A0400")
    void applyRefund_rejectsAmountOverCredits() {
        SysAiBilling billing = new SysAiBilling();
        billing.setId(3L);
        billing.setUserId(1L);
        billing.setCredits(20);
        when(billingMapper.selectById(3L)).thenReturn(billing);
        AiRefundCreateForm form = new AiRefundCreateForm();
        form.setBillingId(3L);
        form.setAmount(21);
        form.setReason("误扣");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.applyRefund(form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
        }
    }

    @Test
    @DisplayName("退款申请：同一计费记录已有待审核申请抛 A0680")
    void applyRefund_rejectsPendingDuplicate() {
        SysAiBilling billing = new SysAiBilling();
        billing.setId(3L);
        billing.setUserId(1L);
        billing.setCredits(20);
        when(billingMapper.selectById(3L)).thenReturn(billing);
        when(refundMapper.selectOne(any())).thenReturn(new SysAiRefund());
        AiRefundCreateForm form = new AiRefundCreateForm();
        form.setBillingId(3L);
        form.setAmount(10);
        form.setReason("误扣");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.applyRefund(form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.AI_REFUND_ALREADY_EXISTS.getCode());
        }
        verify(refundMapper, never()).insert(any());
    }

    @Test
    @DisplayName("退款审核：非待审核状态抛 A0681")
    void auditRefund_rejectsAlreadyAudited() {
        SysAiRefund refund = new SysAiRefund();
        refund.setId(5L);
        refund.setStatus(2);
        when(refundMapper.selectById(5L)).thenReturn(refund);
        AiRefundAuditForm form = new AiRefundAuditForm();
        form.setApproved(true);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.auditRefund(5L, form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.REFUND_AUDIT_FAILED.getCode());
        }
    }

    @Test
    @DisplayName("退款审核通过：余额 INCRBY + MySQL CAS + 写流水，状态置 2")
    void auditRefund_approveRefundsBalance() {
        SysAiRefund refund = new SysAiRefund();
        refund.setId(5L);
        refund.setUserId(9L);
        refund.setBillingId(3L);
        refund.setAmount(20);
        refund.setReason("误扣");
        refund.setStatus(1);
        SysAiRefund refreshed = new SysAiRefund();
        refreshed.setId(5L);
        refreshed.setUserId(9L);
        refreshed.setBillingId(3L);
        refreshed.setAmount(20);
        refreshed.setStatus(2);
        when(refundMapper.selectById(5L)).thenReturn(refund, refreshed);
        when(billingMapper.selectById(3L)).thenReturn(new SysAiBilling());
        when(refundMapper.selectCount(any())).thenReturn(0L);
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(ledgerMapper.getCreditsBalanceAndVersion(9L)).thenReturn(credits(100));
        when(ledgerMapper.increaseBalanceCas(eq(9L), any(), any())).thenReturn(1);
        when(valueOperations.get("ai:balance:9")).thenReturn("100");

        AiRefundAuditForm form = new AiRefundAuditForm();
        form.setApproved(true);
        form.setAuditRemark("核实误扣");
        AiRefundVO vo;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);

            vo = service.auditRefund(5L, form);
        }

        assertThat(vo.getStatus()).isEqualTo(2);
        assertThat(refund.getStatus()).isEqualTo(2);
        assertThat(refund.getAuditorId()).isEqualTo(7L);
        verify(valueOperations).increment("ai:balance:9", 20);
        verify(stringRedisTemplate).delete("ai:arrears:9");
        ArgumentCaptor<SysAiCreditLog> logCaptor = ArgumentCaptor.forClass(SysAiCreditLog.class);
        verify(creditLogMapper).insert(logCaptor.capture());
        assertThat(logCaptor.getValue().getSource()).isEqualTo("refund");
        assertThat(logCaptor.getValue().getRelatedId()).isEqualTo(3L);
        assertThat(logCaptor.getValue().getBalanceAfter()).isEqualTo(100L);
        verify(refundMapper).updateById(refund);
    }

    @Test
    @DisplayName("退款审核通过：原计费记录已退款抛 A0681（防重复回补）")
    void auditRefund_rejectsDuplicatedApproved() {
        SysAiRefund refund = new SysAiRefund();
        refund.setId(5L);
        refund.setBillingId(3L);
        refund.setAmount(20);
        refund.setStatus(1);
        when(refundMapper.selectById(5L)).thenReturn(refund);
        when(billingMapper.selectById(3L)).thenReturn(new SysAiBilling());
        when(refundMapper.selectCount(any())).thenReturn(1L);
        AiRefundAuditForm form = new AiRefundAuditForm();
        form.setApproved(true);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);

            assertThatThrownBy(() -> service.auditRefund(5L, form))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.REFUND_AUDIT_FAILED.getCode());
        }
        verify(creditLogMapper, never()).insert(any());
    }

    // ── 余额与配额 ──────────────────────────────────────────

    @Test
    @DisplayName("手动调整：积分为 0/非整数抛 A0400，用户不存在抛 A0401")
    void adjustCredits_validatesAmountAndUser() {
        AiBillingAdjustForm zero = new AiBillingAdjustForm();
        zero.setUserId(9L);
        zero.setAmount(BigDecimal.ZERO);
        zero.setReason("补偿");
        AiBillingAdjustForm fractional = new AiBillingAdjustForm();
        fractional.setUserId(9L);
        fractional.setAmount(new BigDecimal("1.5"));
        fractional.setReason("补偿");
        AiBillingAdjustForm unknown = new AiBillingAdjustForm();
        unknown.setUserId(9L);
        unknown.setAmount(BigDecimal.TEN);
        unknown.setReason("补偿");
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.adjustCredits(zero))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
            assertThatThrownBy(() -> service.adjustCredits(fractional))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
            assertThatThrownBy(() -> service.adjustCredits(unknown))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
    }

    @Test
    @DisplayName("手动调整：Redis INCRBY + MySQL CAS + 清欠费 + 写流水，返回账号视图")
    void adjustCredits_writesRedisAndLedger() {
        AiBillingAdjustForm form = new AiBillingAdjustForm();
        form.setUserId(9L);
        form.setAmount(BigDecimal.valueOf(50));
        form.setReason("客服补偿");
        when(userMapper.selectById(9L)).thenReturn(new SysUser());
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(ledgerMapper.getCreditsBalanceAndVersion(9L)).thenReturn(credits(100));
        when(ledgerMapper.increaseBalanceCas(eq(9L), any(), any())).thenReturn(1);
        when(valueOperations.get("ai:balance:9")).thenReturn("100");
        when(valueOperations.get("ai:quota:daily:9:" + LocalDate.now(BILLING_ZONE)
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd")))).thenReturn(null);
        when(valueOperations.get("ai:quota:monthly:9:" + LocalDate.now(BILLING_ZONE)
                .format(DateTimeFormatter.ofPattern("yyyy-MM")))).thenReturn(null);
        when(memberMapper.selectOne(any())).thenReturn(null);
        when(stringRedisTemplate.hasKey("ai:arrears:9")).thenReturn(true);

        AiBalanceVO vo;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            vo = service.adjustCredits(form);
        }

        assertThat(vo.getCreditsBalance()).isEqualByComparingTo("100");
        assertThat(vo.getDailyLimit()).isZero();
        assertThat(vo.getMonthlyLimit()).isZero();
        assertThat(vo.getArrearsStatus()).isTrue();
        verify(valueOperations).increment("ai:balance:9", 50);
        verify(stringRedisTemplate).delete("ai:arrears:9");
        ArgumentCaptor<SysAiCreditLog> logCaptor = ArgumentCaptor.forClass(SysAiCreditLog.class);
        verify(creditLogMapper).insert(logCaptor.capture());
        assertThat(logCaptor.getValue().getSource()).isEqualTo("admin_adjust");
        assertThat(logCaptor.getValue().getOperatorId()).isEqualTo(1L);
        assertThat(logCaptor.getValue().getReason()).isEqualTo("客服补偿");
    }

    @Test
    @DisplayName("余额缓存为非整数坏值时删除缓存并回源 MySQL 整数化回填")
    void buildBalance_resetsBrokenCacheValue() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("ai:balance:9")).thenReturn("100.00");
        when(ledgerMapper.getCreditsBalanceAndVersion(9L))
                .thenReturn(credits(100));
        when(valueOperations.get("ai:quota:daily:9:" + LocalDate.now(BILLING_ZONE)
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd")))).thenReturn(null);
        when(valueOperations.get("ai:quota:monthly:9:" + LocalDate.now(BILLING_ZONE)
                .format(DateTimeFormatter.ofPattern("yyyy-MM")))).thenReturn(null);
        lenient().when(memberMapper.selectOne(any())).thenReturn(null);
        lenient().when(stringRedisTemplate.hasKey("ai:arrears:9")).thenReturn(false);

        AiBalanceVO vo;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(9L);

            vo = service.getBalance(null);
        }

        assertThat(vo.getCreditsBalance()).isEqualByComparingTo("100");
        verify(stringRedisTemplate).delete("ai:balance:9");
        verify(valueOperations).set(eq("ai:balance:9"), eq("100"), any(Duration.class));
    }

    @Test
    @DisplayName("查询他人数据：无 ai:billing:stat 抛 A0301")
    void getBalance_othersRequiresStatPermission() {
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());

            assertThatThrownBy(() -> service.getBalance(2L))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.ACCESS_UNAUTHORIZED.getCode());
        }
    }

    // ── 统计与成本 ──────────────────────────────────────────

    @Test
    @DisplayName("管理员统计：非法 groupBy 抛 A0400，缓存命中率按 chat 输入口径计算")
    void getStats_validatesDimensionAndComputesHitRate() {
        AiBillingStatQuery invalid = new AiBillingStatQuery();
        invalid.setGroupBy("week");
        assertThatThrownBy(() -> service.getStats(invalid))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.PARAM_ERROR.getCode());

        AiBillingStatRead row = new AiBillingStatRead();
        row.setDimension("gpt-4o");
        row.setTotalCredits(500L);
        row.setTotalInputTokens(1000L);
        row.setTotalOutputTokens(200L);
        row.setChatCachedTokens(250L);
        row.setCreditsSaved(30L);
        row.setDegradationCount(2L);
        when(insightMapper.statsByDimension(eq("model"), any(), any(), any(), any(), any()))
                .thenReturn(List.of(row));
        AiBillingStatQuery query = new AiBillingStatQuery();
        query.setGroupBy("model");

        List<AiBillingStatVO> stats = service.getStats(query);

        assertThat(stats).hasSize(1);
        assertThat(stats.get(0).getCacheHitRate()).isEqualTo(0.25);
        assertThat(stats.get(0).getTotalCredits()).isEqualTo(500);
        assertThat(stats.get(0).getDegradationCount()).isEqualTo(2);
    }

    @Test
    @DisplayName("管理员统计：脏 groupBy（空串/大小写/中文/带空格/注入串）一律 A0400 且不触达 mapper")
    void getStats_rejectsDirtyDimensions() {
        for (String dirty : List.of("", "USER", "bill_type", "日", "user ", "model;drop")) {
            AiBillingStatQuery query = new AiBillingStatQuery();
            query.setGroupBy(dirty);
            assertThatThrownBy(() -> service.getStats(query))
                    .as("groupBy=%s", dirty)
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.PARAM_ERROR.getCode());
        }
        verify(insightMapper, never()).statsByDimension(any(), any(), any(), any(), any(), any());
    }

    @Test
    @DisplayName("成本统计 overall：收入按订单实收（分转元），输出官方与 AI 参考双口径毛利")
    void getCostStats_overallReturnsTwoMetrics() {
        when(insightMapper.sumCost(any(), any())).thenReturn(new BigDecimal("40.00"));
        when(insightMapper.sumPaidOrderByPackageType(any(), any()))
                .thenReturn(List.of(orderIncome("credit", 100_000L), orderIncome("vip", 50_000L)));

        List<AiCostStatVO> stats = service.getCostStats(null, null, "overall", null, null);

        assertThat(stats).hasSize(2);
        AiCostStatVO overall = stats.get(0);
        assertThat(overall.getMetric()).isEqualTo("overall");
        assertThat(overall.getRevenue()).isEqualTo(1500.0);
        assertThat(overall.getCost()).isEqualTo(40.0);
        assertThat(overall.getProfit()).isEqualTo(1460.0);
        assertThat(overall.getProfitRate()).isEqualTo(0.9733);
        AiCostStatVO ai = stats.get(1);
        assertThat(ai.getMetric()).isEqualTo("ai");
        assertThat(ai.getRevenue()).isEqualTo(1150.0);
    }

    @Test
    @DisplayName("成本统计分组：仅输出维度值与成本，不携带收入/毛利/口径")
    void getCostStats_groupedReturnsCostOnly() {
        AiCostStatRead row = new AiCostStatRead();
        row.setDimension("gpt-4o");
        row.setCost(new BigDecimal("12.345"));
        when(insightMapper.sumCostGroupBy(eq("model"), any(), any(), any(), any())).thenReturn(List.of(row));

        List<AiCostStatVO> stats = service.getCostStats(null, null, "model", "gpt-4o", null);

        assertThat(stats).hasSize(1);
        assertThat(stats.get(0).getDimension()).isEqualTo("gpt-4o");
        assertThat(stats.get(0).getCost()).isEqualTo(12.35);
        assertThat(stats.get(0).getRevenue()).isNull();
        assertThat(stats.get(0).getProfit()).isNull();
        assertThat(stats.get(0).getMetric()).isNull();
    }

    @Test
    @DisplayName("成本统计：非法 groupBy 抛 A0400")
    void getCostStats_rejectsUnknownGroupBy() {
        assertThatThrownBy(() -> service.getCostStats(null, null, "agent", null, null))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.PARAM_ERROR.getCode());
    }

    @Test
    @DisplayName("对账导入：统计非空行数")
    void importReconcile_countsNonBlankLines() {
        AiReconcileImportForm form = new AiReconcileImportForm();
        form.setContent("a,b\n\n   \nc,d\n");

        Map<String, Object> result = service.importReconcile(form);

        assertThat(result).containsEntry("imported", 2);
    }

    @Test
    @DisplayName("成本单价新增：版本号递增并落档位明细")
    void createCost_assignsVersionAndDetails() {
        AiModelCostForm form = new AiModelCostForm();
        form.setModelId("gpt-4o");
        form.setProviderId(3L);
        AiModelCostForm.Detail detail = new AiModelCostForm.Detail();
        detail.setTokenType("input");
        detail.setTimeSlot("peak");
        detail.setMinTokens(0L);
        detail.setUnitPrice(new BigDecimal("30"));
        form.setDetails(List.of(detail));
        when(modelCostMapper.nextPriceVersion("gpt-4o", 3L)).thenReturn(3);
        when(modelCostMapper.insert(any(SysAiModelCost.class))).thenAnswer(invocation -> {
            SysAiModelCost cost = invocation.getArgument(0);
            cost.setId(11L);
            return 1;
        });

        AiModelCostVO vo = service.createCost(form);

        assertThat(vo.getId()).isEqualTo(11L);
        assertThat(vo.getPriceVersion()).isEqualTo(3);
        assertThat(vo.getCurrency()).isEqualTo("CNY");
        assertThat(vo.getStatus()).isEqualTo(1);
        assertThat(vo.getDetails()).hasSize(1);
        assertThat(vo.getDetails().get(0).getPriceId()).isEqualTo(11L);
        ArgumentCaptor<SysAiModelCostDetail> captor = ArgumentCaptor.forClass(SysAiModelCostDetail.class);
        verify(modelCostDetailMapper).insert(captor.capture());
        assertThat(captor.getValue().getTokenType()).isEqualTo("input");
        assertThat(captor.getValue().getTimeSlot()).isEqualTo("peak");
    }

    @Test
    @DisplayName("成本单价删除：不存在抛 A0401，存在则主表与档位明细一并软删")
    void deleteCost_softDeletesMainAndDetails() {
        when(modelCostMapper.selectById(9L)).thenReturn(null, new SysAiModelCost());
        assertThatThrownBy(() -> service.deleteCost(9L))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());

        service.deleteCost(9L);

        verify(modelCostMapper).deleteById(9L);
        verify(modelCostDetailMapper).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("计费明细分页：按用户过滤并回填最新退款状态")
    void listRecords_fillsRefundStatus() {
        SysAiBilling billing = new SysAiBilling();
        billing.setId(7L);
        billing.setUserId(1L);
        billing.setCredits(12);
        IPage<SysAiBilling> page = new com.baomidou.mybatisplus.extension.plugins.pagination.Page<>(1, 20, 1);
        page.setRecords(List.of(billing));
        when(billingMapper.selectPage(any(), any())).thenReturn(page);
        SysAiRefund refund = new SysAiRefund();
        refund.setBillingId(7L);
        refund.setStatus(2);
        SysAiRefund older = new SysAiRefund();
        older.setBillingId(7L);
        older.setStatus(3);
        when(refundMapper.selectList(any())).thenReturn(List.of(older, refund));
        com.pei.dehaze.model.query.AiBillingRecordQuery query =
                new com.pei.dehaze.model.query.AiBillingRecordQuery();
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            IPage<com.pei.dehaze.model.vo.AiBillingRecordVO> result = service.listRecords(query);

            assertThat(result.getTotal()).isEqualTo(1);
            assertThat(result.getRecords().get(0).getRefundStatus()).isEqualTo(2);
        }
    }

    private static AiUserCreditsRead credits(long balance) {
        AiUserCreditsRead read = new AiUserCreditsRead();
        read.setCreditsBalance(BigDecimal.valueOf(balance));
        read.setCreditsVersion(1);
        return read;
    }

    private static AiOrderIncomeRead orderIncome(String packageType, long amount) {
        AiOrderIncomeRead read = new AiOrderIncomeRead();
        read.setPackageType(packageType);
        read.setAmount(amount);
        return read;
    }
}
