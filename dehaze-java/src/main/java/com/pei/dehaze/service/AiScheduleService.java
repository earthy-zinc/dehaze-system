package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiScheduleMapper;
import com.pei.dehaze.mapper.SysAiScheduleRunMapper;
import com.pei.dehaze.model.entity.SysAiSchedule;
import com.pei.dehaze.model.entity.SysAiScheduleRun;
import com.pei.dehaze.model.form.AiScheduleCreateForm;
import com.pei.dehaze.model.form.AiScheduleUpdateForm;
import com.pei.dehaze.model.query.AiSchedulePageQuery;
import com.pei.dehaze.model.vo.AiNextTimesVO;
import com.pei.dehaze.model.vo.AiScheduleHistoryVO;
import com.pei.dehaze.model.vo.AiScheduleRunSummaryVO;
import com.pei.dehaze.model.vo.AiScheduleVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.support.CronExpression;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AI 定时任务服务（F-M08-009）。
 *
 * <p>对齐 dehaze-python {@code scheduled_task_service}：VIP2+ 才可创建、单用户上限 20、
 * Cron 归一化与校验（标准 5 位表达式 + daily@/weekly@/monthly@ 频率标识）、下次触发时间按任务时区计算、
 * 启用时若处于熔断停用则重置熔断计数。手动触发执行（/run）走转发域。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiScheduleService {

    private static final int MAX_SCHEDULES_PER_USER = 20;

    private static final String DEFAULT_TIMEZONE = "Asia/Shanghai";

    private static final Map<String, Integer> LEVEL_MAP = Map.of(
            "level_0", 0, "level_1", 1, "level_2", 2, "level_3", 3);

    private static final Map<String, Integer> WEEKDAY_ALIAS = Map.of(
            "mon", 1, "tue", 2, "wed", 3, "thu", 4, "fri", 5, "sat", 6, "sun", 0);

    private static final Map<String, String> WEEKDAY_LABEL = Map.of(
            "0", "日", "1", "一", "2", "二", "3", "三", "4", "四", "5", "五", "6", "六", "7", "日");

    private final SysAiScheduleMapper scheduleMapper;

    private final SysAiScheduleRunMapper scheduleRunMapper;

    private final AiInsightMapper insightMapper;

    @Transactional
    public AiScheduleVO create(Long userId, AiScheduleCreateForm form) {
        ensureVip2(userId);
        if (scheduleMapper.countByUserId(userId) >= MAX_SCHEDULES_PER_USER) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW,
                    "定时任务数量已达上限(" + MAX_SCHEDULES_PER_USER + "个)");
        }
        String timezone = form.getTimezone() == null || form.getTimezone().isBlank()
                ? DEFAULT_TIMEZONE : form.getTimezone();
        String cron = normalizeCron(form.getCron());
        parseCron(cron);
        SysAiSchedule task = new SysAiSchedule();
        task.setUserId(userId);
        task.setName(form.getName().trim());
        task.setCron(cron);
        task.setTimezone(timezone);
        task.setInput(form.getInput());
        task.setOutput(form.getOutput());
        task.setEnabled(1);
        task.setStatus(1);
        task.setCircuitStreak(0);
        task.setNextTriggerTime(computeNextTrigger(cron, timezone));
        scheduleMapper.insert(task);
        return toVO(task);
    }

    @Transactional
    public AiScheduleVO update(Long userId, Long scheduleId, AiScheduleUpdateForm form) {
        SysAiSchedule task = getOwned(userId, scheduleId);
        if (form.getName() != null) {
            task.setName(form.getName().trim());
        }
        if (form.getCron() != null) {
            String cron = normalizeCron(form.getCron());
            parseCron(cron);
            task.setCron(cron);
        }
        if (form.getTimezone() != null) {
            task.setTimezone(form.getTimezone());
        }
        if (form.getInput() != null) {
            task.setInput(form.getInput());
        }
        if (form.getOutput() != null) {
            task.setOutput(form.getOutput());
        }
        if (form.getEnabled() != null) {
            task.setEnabled(form.getEnabled());
        }
        if (form.getCron() != null || form.getTimezone() != null || Integer.valueOf(1).equals(form.getEnabled())) {
            task.setNextTriggerTime(computeNextTrigger(task.getCron(), task.getTimezone()));
        }
        scheduleMapper.updateById(task);
        return toVO(task);
    }

    public AiScheduleVO getDetail(Long userId, Long scheduleId) {
        return toVO(getOwned(userId, scheduleId));
    }

    /**
     * 任务列表：启用优先、下次触发时间升序（NULL 排后），批量附加最近一次执行摘要
     */
    public IPage<AiScheduleVO> list(Long userId, AiSchedulePageQuery query) {
        LambdaQueryWrapper<SysAiSchedule> wrapper = new LambdaQueryWrapper<SysAiSchedule>()
                .eq(SysAiSchedule::getUserId, userId)
                .orderByDesc(SysAiSchedule::getEnabled)
                .orderByAsc(SysAiSchedule::getNextTriggerTime)
                .orderByAsc(SysAiSchedule::getId);
        if (query.getKeyword() != null && !query.getKeyword().isBlank()) {
            wrapper.like(SysAiSchedule::getName, query.getKeyword());
        }
        Page<SysAiSchedule> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiSchedule> taskPage = scheduleMapper.selectPage(page, wrapper);
        List<Long> scheduleIds = taskPage.getRecords().stream().map(SysAiSchedule::getId).toList();
        Map<Long, SysAiScheduleRun> latest = new HashMap<>();
        if (!scheduleIds.isEmpty()) {
            for (SysAiScheduleRun run : scheduleRunMapper.selectLatestByScheduleIds(scheduleIds)) {
                latest.put(run.getScheduleId(), run);
            }
        }
        List<AiScheduleVO> records = new ArrayList<>();
        for (SysAiSchedule task : taskPage.getRecords()) {
            AiScheduleVO vo = toVO(task);
            SysAiScheduleRun run = latest.get(task.getId());
            if (run != null) {
                AiScheduleRunSummaryVO summary = new AiScheduleRunSummaryVO();
                summary.setStatus(run.getStatus());
                summary.setSkipReason(run.getSkipReason());
                summary.setCredits(run.getCredits());
                summary.setDurationMs(run.getDurationMs());
                summary.setErrorMsg(run.getErrorMsg());
                summary.setConversationId(run.getConversationId());
                summary.setCreateTime(run.getCreateTime());
                vo.setLastRun(summary);
            }
            records.add(vo);
        }
        Page<AiScheduleVO> result = new Page<>(query.getPageNum(), query.getPageSize(), taskPage.getTotal());
        result.setRecords(records);
        return result;
    }

    /**
     * 启停任务：启用时若处于熔断停用(status=2)则重置熔断计数，并重算下次触发时间
     */
    @Transactional
    public void setEnabled(Long userId, Long scheduleId, Integer enabled) {
        SysAiSchedule task = getOwned(userId, scheduleId);
        if (Integer.valueOf(1).equals(enabled)) {
            if (Integer.valueOf(2).equals(task.getStatus())) {
                scheduleMapper.resetCircuit(scheduleId);
            }
            scheduleMapper.updateNextTrigger(scheduleId, computeNextTrigger(task.getCron(), task.getTimezone()));
        }
        scheduleMapper.setEnabled(scheduleId, enabled);
    }

    @Transactional
    public void delete(Long userId, Long scheduleId) {
        getOwned(userId, scheduleId);
        scheduleMapper.softDelete(scheduleId);
    }

    public IPage<AiScheduleHistoryVO> listHistory(Long userId, Long scheduleId, int pageNum, int pageSize) {
        getOwned(userId, scheduleId);
        Page<SysAiScheduleRun> page = new Page<>(pageNum, pageSize);
        IPage<SysAiScheduleRun> runPage = scheduleRunMapper.selectPageBySchedule(page, scheduleId);
        List<AiScheduleHistoryVO> records = new ArrayList<>();
        for (SysAiScheduleRun run : runPage.getRecords()) {
            AiScheduleHistoryVO vo = new AiScheduleHistoryVO();
            vo.setId(run.getId());
            vo.setScheduleId(run.getScheduleId());
            vo.setStatus(run.getStatus());
            vo.setSkipReason(run.getSkipReason());
            vo.setCredits(run.getCredits());
            vo.setDurationMs(run.getDurationMs());
            vo.setErrorMsg(run.getErrorMsg());
            vo.setConversationId(run.getConversationId());
            vo.setRequestId(run.getRequestId());
            vo.setWindowStart(run.getWindowStart());
            vo.setCreateTime(run.getCreateTime());
            records.add(vo);
        }
        Page<AiScheduleHistoryVO> result = new Page<>(pageNum, pageSize, runPage.getTotal());
        result.setRecords(records);
        return result;
    }

    /**
     * Cron 解释与接下来 N 次触发时间预览（非法 Cron/时区抛参数异常）
     */
    public AiNextTimesVO previewNextTimes(String cron, int count) {
        String normalized = normalizeCron(cron);
        CronExpression expression = parseCron(normalized);
        ZoneId zone = resolveZone(DEFAULT_TIMEZONE);
        List<OffsetDateTime> nextTimes = new ArrayList<>();
        ZonedDateTime cursor = ZonedDateTime.now(zone);
        for (int i = 0; i < count; i++) {
            cursor = expression.next(cursor);
            if (cursor == null) {
                break;
            }
            nextTimes.add(cursor.toOffsetDateTime());
        }
        AiNextTimesVO vo = new AiNextTimesVO();
        vo.setDescription(describeCron(normalized));
        vo.setNextTimes(nextTimes);
        return vo;
    }

    /**
     * 归一化触发规则：daily@HH:MM / weekly@D@HH:MM / monthly@D@HH:MM → 标准 5 位 Cron
     */
    String normalizeCron(String raw) {
        String text = raw == null ? "" : raw.trim();
        if (!text.contains("@")) {
            return text;
        }
        String[] parts = text.split("@");
        if (parts.length < 2 || parts.length > 3
                || !List.of("daily", "weekly", "monthly").contains(parts[0])) {
            return text;
        }
        String[] hm = parts[parts.length - 1].split(":");
        int hour;
        int minute;
        try {
            if (hm.length != 2) {
                throw new NumberFormatException();
            }
            hour = Integer.parseInt(hm[0].trim());
            minute = Integer.parseInt(hm[1].trim());
        } catch (NumberFormatException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "触发规则时间格式非法: " + raw);
        }
        if (hour < 0 || hour > 23 || minute < 0 || minute > 59) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "触发规则时间超出范围: " + raw);
        }
        if ("daily".equals(parts[0])) {
            return minute + " " + hour + " * * *";
        }
        String day = parts[1].trim().toLowerCase();
        if ("weekly".equals(parts[0])) {
            Integer weekday = WEEKDAY_ALIAS.get(day);
            if (weekday == null) {
                try {
                    weekday = Math.floorMod(Integer.parseInt(day), 7);
                } catch (NumberFormatException e) {
                    return text;
                }
            }
            return minute + " " + hour + " * * " + weekday;
        }
        int dayOfMonth;
        try {
            dayOfMonth = Integer.parseInt(day);
        } catch (NumberFormatException e) {
            return text;
        }
        if (dayOfMonth < 1 || dayOfMonth > 31) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "触发规则日期超出范围: " + raw);
        }
        return minute + " " + hour + " " + dayOfMonth + " * *";
    }

    private CronExpression parseCron(String cron) {
        String normalized = cron.replaceAll("\\s+", " ").trim();
        String[] fields = normalized.split(" ");
        if (fields.length != 5) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "Cron 表达式非法: " + cron);
        }
        // 5 位 UNIX Cron（分 时 日 月 周）转 Spring 6 位（补秒位），星期名归一为数字
        String weekday = fields[4].toLowerCase().replace("mon", "1").replace("tue", "2")
                .replace("wed", "3").replace("thu", "4").replace("fri", "5")
                .replace("sat", "6").replace("sun", "0");
        String springCron = "0 " + fields[0] + " " + fields[1] + " " + fields[2] + " "
                + fields[3] + " " + weekday;
        try {
            return CronExpression.parse(springCron);
        } catch (IllegalArgumentException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "Cron 表达式非法: " + cron);
        }
    }

    private ZoneId resolveZone(String timezone) {
        try {
            return ZoneId.of(timezone);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "任务时区非法: " + timezone);
        }
    }

    /**
     * 计算下次触发时间（按任务时区计算后转本地时间落库，用于排序与扫描）
     */
    private LocalDateTime computeNextTrigger(String cron, String timezone) {
        ZoneId zone = resolveZone(timezone);
        ZonedDateTime next = parseCron(cron).next(ZonedDateTime.now(zone));
        if (next == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "Cron 表达式无法计算下次触发时间: " + cron);
        }
        return next.withZoneSameInstant(ZoneId.systemDefault()).toLocalDateTime();
    }

    /**
     * Cron 人类可读描述：覆盖常用频率，无法归类时回退原始表达式
     */
    private String describeCron(String cron) {
        try {
            String[] parts = cron.split("\\s+");
            if (parts.length != 5) {
                return cron;
            }
            String minute = parts[0];
            String hour = parts[1];
            String day = parts[2];
            String month = parts[3];
            String weekday = parts[4];
            if ("*".equals(minute) && "*".equals(hour) && "*".equals(day) && "*".equals(month)
                    && "*".equals(weekday)) {
                return "每分钟";
            }
            if ("*".equals(hour) && "*".equals(day) && "*".equals(month) && "*".equals(weekday)) {
                return "0".equals(minute) ? "每小时整点" : "每小时 " + fmtMinute(minute) + "分";
            }
            if (!"*".equals(hour) && "*".equals(day) && "*".equals(month) && "*".equals(weekday)) {
                return "每天 " + fmtHour(hour) + "点" + fmtMinute(minute) + "分";
            }
            if ("*".equals(day) && "*".equals(month) && !"*".equals(weekday)) {
                List<String> days = new ArrayList<>();
                for (String value : weekday.split(",")) {
                    if (WEEKDAY_LABEL.containsKey(value)) {
                        days.add("周" + WEEKDAY_LABEL.get(value));
                    }
                }
                return "每周" + String.join("、", days) + " " + timePart(hour, minute);
            }
            if ("*".equals(month) && "*".equals(weekday) && !"*".equals(day)) {
                List<String> days = new ArrayList<>();
                for (String value : day.split(",")) {
                    days.add(value + "号");
                }
                return "每月" + String.join("、", days) + " " + timePart(hour, minute);
            }
            if (!"*".equals(month) && "*".equals(weekday) && !"*".equals(day)) {
                List<String> months = new ArrayList<>();
                for (String value : month.split(",")) {
                    months.add(value + "月");
                }
                List<String> days = new ArrayList<>();
                for (String value : day.split(",")) {
                    days.add(value + "号");
                }
                return "每年" + String.join("、", months) + String.join("、", days) + " "
                        + timePart(hour, minute);
            }
        } catch (Exception e) {
            log.debug("Cron 描述生成失败，回退原始表达式: {}", cron);
        }
        return "Cron(" + cron + ")";
    }

    private String timePart(String hour, String minute) {
        return "*".equals(hour) ? "每小时" + fmtMinute(minute) + "分" : fmtHour(hour) + "点" + fmtMinute(minute) + "分";
    }

    private String fmtHour(String hour) {
        return String.format("%02d", Integer.parseInt(hour));
    }

    private String fmtMinute(String minute) {
        return "*".equals(minute) ? "00" : String.format("%02d", Integer.parseInt(minute));
    }

    /**
     * 定时调度功能仅 VIP2+ 可用（无会员记录视为 level_0）
     */
    private void ensureVip2(Long userId) {
        String levelCode = insightMapper.selectMemberLevelCode(userId);
        int level = levelCode == null ? 0 : LEVEL_MAP.getOrDefault(levelCode, 0);
        if (level < 2) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW,
                    "定时调度功能需 VIP2 及以上会员，请升级会员后使用");
        }
    }

    private SysAiSchedule getOwned(Long userId, Long scheduleId) {
        SysAiSchedule task = scheduleMapper.selectOne(new LambdaQueryWrapper<SysAiSchedule>()
                .eq(SysAiSchedule::getId, scheduleId));
        if (task == null || !task.getUserId().equals(userId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "定时任务不存在");
        }
        return task;
    }

    private AiScheduleVO toVO(SysAiSchedule task) {
        AiScheduleVO vo = new AiScheduleVO();
        vo.setId(task.getId());
        vo.setUserId(task.getUserId());
        vo.setName(task.getName());
        vo.setCron(task.getCron());
        vo.setTimezone(task.getTimezone());
        vo.setInput(task.getInput());
        vo.setOutput(task.getOutput());
        vo.setEnabled(task.getEnabled());
        vo.setStatus(task.getStatus());
        vo.setCircuitStreak(task.getCircuitStreak());
        vo.setNextTriggerTime(task.getNextTriggerTime());
        vo.setCreateTime(task.getCreateTime());
        return vo;
    }
}
