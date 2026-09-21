package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiScheduleMapper;
import com.pei.dehaze.mapper.SysAiScheduleRunMapper;
import com.pei.dehaze.model.entity.SysAiSchedule;
import com.pei.dehaze.model.form.AiScheduleCreateForm;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.function.Executable;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AI 定时任务服务单测：会员门槛、数量上限、频率标识归一化、Cron 校验与下次触发时间、熔断重置。
 *
 * <p>对齐 dehaze-python {@code scheduled_task_service}：VIP2+ 才可用、单用户上限 20、
 * 支持 {@code daily@HH:MM} / {@code weekly@D@HH:MM} / {@code monthly@D@HH:MM} 归一为标准 5 位 Cron。
 */
@DisplayName("AiScheduleService 定时任务")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiScheduleServiceTest {

    private static final Long USER_ID = 3L;

    @Mock
    private SysAiScheduleMapper scheduleMapper;
    @Mock
    private SysAiScheduleRunMapper scheduleRunMapper;
    @Mock
    private AiInsightMapper insightMapper;

    @InjectMocks
    private AiScheduleService service;

    private void vip2() {
        when(insightMapper.selectMemberLevelCode(USER_ID)).thenReturn("level_2");
    }

    private SysAiSchedule task(Long id, Long userId, int status) {
        SysAiSchedule task = new SysAiSchedule();
        task.setId(id);
        task.setUserId(userId);
        task.setName("每日去雾");
        task.setCron("30 9 * * *");
        task.setTimezone("Asia/Shanghai");
        task.setEnabled(1);
        task.setStatus(status);
        task.setCircuitStreak(0);
        return task;
    }

    private void assertBizError(ResultCode expected, Executable action) {
        assertThat(assertThrows(BusinessException.class, action).getResultCode()).isEqualTo(expected);
    }

    @Test
    @DisplayName("频率标识归一化为标准 5 位 Cron（daily/weekly/monthly）")
    void normalizeCronExpandsFrequencyAliases() {
        assertThat(service.normalizeCron("daily@09:30")).isEqualTo("30 9 * * *");
        assertThat(service.normalizeCron("weekly@mon@08:00")).isEqualTo("0 8 * * 1");
        assertThat(service.normalizeCron("weekly@SUN@08:05")).isEqualTo("5 8 * * 0");
        assertThat(service.normalizeCron("weekly@7@00:00")).isEqualTo("0 0 * * 0");
        assertThat(service.normalizeCron("monthly@15@06:00")).isEqualTo("0 6 15 * *");
    }

    @Test
    @DisplayName("已是标准 Cron 或无法识别的表达式按原样返回")
    void normalizeCronKeepsPlainExpression() {
        assertThat(service.normalizeCron("0 9 * * *")).isEqualTo("0 9 * * *");
        assertThat(service.normalizeCron("hourly@9:00")).isEqualTo("hourly@9:00");
    }

    @Test
    @DisplayName("频率标识时间非法报 A0400（格式/范围）")
    void normalizeCronRejectsInvalidTime() {
        assertBizError(ResultCode.PARAM_ERROR, () -> service.normalizeCron("daily@9"));
        assertBizError(ResultCode.PARAM_ERROR, () -> service.normalizeCron("daily@25:00"));
        assertBizError(ResultCode.PARAM_ERROR, () -> service.normalizeCron("monthly@32@06:00"));
    }

    @Test
    @DisplayName("下次触发时间预览：返回人类可读描述与递增的触发时刻")
    void previewNextTimesReturnsIncreasingTimes() {
        var preview = service.previewNextTimes("daily@09:30", 3);

        assertThat(preview.getDescription()).isEqualTo("每天 09点30分");
        assertThat(preview.getNextTimes()).hasSize(3);
        assertThat(preview.getNextTimes().get(0)).isBefore(preview.getNextTimes().get(1));
        assertThat(preview.getNextTimes().get(1)).isBefore(preview.getNextTimes().get(2));
    }

    @Test
    @DisplayName("非法 Cron 预览报 A0400")
    void previewNextTimesRejectsInvalidCron() {
        assertBizError(ResultCode.PARAM_ERROR, () -> service.previewNextTimes("not a cron", 3));
    }

    @Test
    @DisplayName("创建任务：非 VIP2 报 A0503")
    void createRequiresVip2() {
        when(insightMapper.selectMemberLevelCode(USER_ID)).thenReturn("level_1");
        AiScheduleCreateForm form = new AiScheduleCreateForm();
        form.setName("每日去雾");
        form.setCron("daily@09:30");

        assertBizError(ResultCode.OPERATION_NOT_ALLOW, () -> service.create(USER_ID, form));
        verify(scheduleMapper, never()).insert(any(SysAiSchedule.class));
    }

    @Test
    @DisplayName("创建任务：达到单用户上限 20 报 A0502")
    void createRejectsWhenLimitReached() {
        vip2();
        when(scheduleMapper.countByUserId(USER_ID)).thenReturn(20L);
        AiScheduleCreateForm form = new AiScheduleCreateForm();
        form.setName("每日去雾");
        form.setCron("daily@09:30");

        assertBizError(ResultCode.DATA_STATE_NOT_ALLOW, () -> service.create(USER_ID, form));
        verify(scheduleMapper, never()).insert(any(SysAiSchedule.class));
    }

    @Test
    @DisplayName("创建任务：默认时区 Asia/Shanghai、启用状态并预置下次触发时间")
    void createFillsDefaultsAndNextTrigger() {
        vip2();
        when(scheduleMapper.countByUserId(USER_ID)).thenReturn(0L);
        AiScheduleCreateForm form = new AiScheduleCreateForm();
        form.setName("每日去雾");
        form.setCron("daily@09:30");

        var vo = service.create(USER_ID, form);

        assertThat(vo.getCron()).isEqualTo("30 9 * * *");
        assertThat(vo.getTimezone()).isEqualTo("Asia/Shanghai");
        assertThat(vo.getEnabled()).isEqualTo(1);
        assertThat(vo.getStatus()).isEqualTo(1);
        assertThat(vo.getNextTriggerTime()).isNotNull();
    }

    @Test
    @DisplayName("启停：从熔断停用态重新启用时重置熔断并重算下次触发时间")
    void setEnabledResetsCircuitWhenReenabling() {
        when(scheduleMapper.selectOne(any(Wrapper.class))).thenReturn(task(9L, USER_ID, 2));

        service.setEnabled(USER_ID, 9L, 1);

        verify(scheduleMapper).resetCircuit(9L);
        verify(scheduleMapper).updateNextTrigger(eq(9L), any());
        verify(scheduleMapper).setEnabled(9L, 1);
    }

    @Test
    @DisplayName("启停：停用不重算触发时间")
    void setEnabledDisableSkipsNextTrigger() {
        when(scheduleMapper.selectOne(any(Wrapper.class))).thenReturn(task(9L, USER_ID, 1));

        service.setEnabled(USER_ID, 9L, 0);

        verify(scheduleMapper, never()).updateNextTrigger(any(), any());
        verify(scheduleMapper).setEnabled(9L, 0);
    }

    @Test
    @DisplayName("他人任务不可见（A0401），删除需归属校验")
    void otherUsersTaskIsInvisible() {
        when(scheduleMapper.selectOne(any(Wrapper.class))).thenReturn(task(9L, 99L, 1));

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.delete(USER_ID, 9L));
        verify(scheduleMapper, never()).softDelete(any());
    }

    @Test
    @DisplayName("删除任务：本人任务软删")
    void deleteOwnTaskSoftDeletes() {
        when(scheduleMapper.selectOne(any(Wrapper.class))).thenReturn(task(9L, USER_ID, 1));

        service.delete(USER_ID, 9L);

        verify(scheduleMapper).softDelete(9L);
    }

    @Test
    @DisplayName("时区非法报 A0400")
    void invalidTimezoneRejected() {
        vip2();
        when(scheduleMapper.countByUserId(USER_ID)).thenReturn(0L);
        AiScheduleCreateForm form = new AiScheduleCreateForm();
        form.setName("每日去雾");
        form.setCron("daily@09:30");
        form.setTimezone("Mars/Olympus");

        assertBizError(ResultCode.PARAM_ERROR, () -> service.create(USER_ID, form));
    }

    @Test
    @DisplayName("常见频率的描述文案覆盖：每分钟/每小时整点/每天/每周/每月")
    void describeCommonFrequencies() {
        assertThat(service.previewNextTimes("* * * * *", 1).getDescription()).isEqualTo("每分钟");
        assertThat(service.previewNextTimes("0 * * * *", 1).getDescription()).isEqualTo("每小时整点");
        assertThat(service.previewNextTimes("15 * * * *", 1).getDescription()).isEqualTo("每小时 15分");
        assertThat(service.previewNextTimes("weekly@mon@08:00", 1).getDescription()).contains("每周");
        assertThat(service.previewNextTimes("monthly@1@08:00", 1).getDescription()).contains("每月");
    }

    @Test
    @DisplayName("预览条数为 0 时不计算触发时刻")
    void previewZeroCountReturnsEmptyList() {
        assertThat(service.previewNextTimes("daily@09:30", 0).getNextTimes()).isEmpty();
    }

    @Test
    @DisplayName("列表仅返回本人任务并按启用优先排序")
    void listScopesToCurrentUser() {
        when(scheduleMapper.selectPage(any(), any(Wrapper.class))).thenReturn(new com.baomidou.mybatisplus.extension.plugins.pagination.Page<>(1, 10, 0));

        assertThat(service.list(USER_ID, new com.pei.dehaze.model.query.AiSchedulePageQuery()).getRecords()).isEmpty();
        verify(scheduleRunMapper, never()).selectLatestByScheduleIds(any());
    }

    @Test
    @DisplayName("每年规则可归类；无法归类的分钟列表回退为 Cron(...) 原始表达式")
    void describeFallsBackToRawExpression() {
        assertThat(service.previewNextTimes("0 0 1 1 *", 1).getDescription()).startsWith("每年");
        assertThat(service.previewNextTimes("1,2 * * * *", 1).getDescription())
                .isEqualTo("Cron(1,2 * * * *)");
    }
}
