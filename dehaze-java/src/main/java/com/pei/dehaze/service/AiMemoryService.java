package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiMemoryMapper;
import com.pei.dehaze.model.entity.SysAiMemory;
import com.pei.dehaze.model.form.AiMemoryCreateForm;
import com.pei.dehaze.model.form.AiMemoryUpdateForm;
import com.pei.dehaze.model.vo.AiMemoryVO;
import com.pei.dehaze.service.impl.AiKbIndexClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * AI 长期记忆服务。
 *
 * <p>对齐 dehaze-python {@code ai_memory_service}：软删 + 30 天恢复窗口、归档只读视图、
 * 检索命中重激活（access_count+1 / 刷新衰减计时器 / importance+5）、批量清空与恢复需二次确认。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiMemoryService {

    private static final int RECOVERY_WINDOW_DAYS = 30;

    private static final DateTimeFormatter TIME_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final SysAiMemoryMapper memoryMapper;

    private final AuditLogService auditLogService;

    private final ObjectMapper objectMapper;

    private final AiKbIndexClient esDocClient;

    /**
     * 记忆导出结果（控制器直接写响应体）
     */
    public record MemoryExport(String filename, String contentType, String content) {
    }

    public IPage<AiMemoryVO> list(Long userId, int pageNum, int pageSize, String memoryType, String source) {
        LambdaQueryWrapper<SysAiMemory> wrapper = new LambdaQueryWrapper<SysAiMemory>()
                .eq(SysAiMemory::getUserId, userId)
                .eq(SysAiMemory::getStatus, 1)
                .eq(SysAiMemory::getArchived, 0)
                .eq(memoryType != null, SysAiMemory::getMemoryType, memoryType)
                .eq(source != null, SysAiMemory::getSource, source)
                .orderByDesc(SysAiMemory::getImportance)
                .orderByDesc(SysAiMemory::getCreateTime);
        Page<SysAiMemory> page = new Page<>(pageNum, pageSize);
        IPage<SysAiMemory> memoryPage = memoryMapper.selectPage(page, wrapper);
        return pageOf(page, memoryPage.getRecords().stream().map(this::toVO).toList(), memoryPage.getTotal());
    }

    public IPage<AiMemoryVO> listArchived(Long userId, int pageNum, int pageSize, String memoryType) {
        LambdaQueryWrapper<SysAiMemory> wrapper = new LambdaQueryWrapper<SysAiMemory>()
                .eq(SysAiMemory::getUserId, userId)
                .eq(SysAiMemory::getArchived, 1)
                .eq(memoryType != null, SysAiMemory::getMemoryType, memoryType)
                .orderByDesc(SysAiMemory::getImportance)
                .orderByDesc(SysAiMemory::getCreateTime);
        Page<SysAiMemory> page = new Page<>(pageNum, pageSize);
        IPage<SysAiMemory> memoryPage = memoryMapper.selectPage(page, wrapper);
        return pageOf(page, memoryPage.getRecords().stream().map(this::toVO).toList(), memoryPage.getTotal());
    }

    @Transactional
    public AiMemoryVO create(Long userId, AiMemoryCreateForm form) {
        SysAiMemory memory = new SysAiMemory();
        memory.setUserId(userId);
        memory.setMemoryType(form.getMemoryType());
        memory.setContent(form.getContent());
        memory.setMetadata(form.getMetadata());
        memory.setImportance(form.getImportance() == null ? 50 : form.getImportance());
        memory.setSource(form.getSource() == null ? "manual" : form.getSource());
        memory.setStatus(1);
        memory.setArchived(0);
        memory.setAccessCount(0);
        memoryMapper.insert(memory);
        return toVO(memory);
    }

    @Transactional
    public AiMemoryVO update(Long memoryId, Long userId, AiMemoryUpdateForm form) {
        SysAiMemory memory = getOwned(memoryId, userId);
        if (form.getContent() != null) {
            memory.setContent(form.getContent());
        }
        if (form.getImportance() != null) {
            memory.setImportance(form.getImportance());
        }
        if (form.getStatus() != null) {
            memory.setStatus(form.getStatus());
        }
        memoryMapper.updateById(memory);
        return toVO(memory);
    }

    @Transactional
    public void delete(Long memoryId, Long userId) {
        SysAiMemory memory = getOwned(memoryId, userId);
        memoryMapper.softDeleteByIds(List.of(memory.getId()), userId);
        // 同步清除 ES 向量文档，否则已删记忆仍被向量检索召回
        esDocClient.deleteMemoryDoc(memory.getId());
    }

    /**
     * 取消归档：恢复注入并刷新衰减计时器。
     *
     * <p>归档是遗忘策略的系统行为，用户侧只提供"取消归档"这一个反向操作（不存在手动归档）；
     * 不刷新 last_accessed_at 时，下次每日归档任务会按旧时点立刻再次归档，用户操作形同无效。
     */
    @Transactional
    public AiMemoryVO unarchive(Long memoryId, Long userId) {
        SysAiMemory memory = getOwned(memoryId, userId);
        if (!Integer.valueOf(1).equals(memory.getArchived())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该记忆未处于归档状态");
        }
        memory.setArchived(0);
        memory.setLastAccessedAt(LocalDateTime.now());
        memoryMapper.updateById(memory);
        return toVO(memory);
    }

    /**
     * 关键词搜索：命中后重激活（访问计数 +1、刷新衰减计时器、重要性 +5）
     */
    @Transactional
    public List<AiMemoryVO> search(Long userId, String keyword, int limit) {
        List<SysAiMemory> memories = memoryMapper.selectList(new LambdaQueryWrapper<SysAiMemory>()
                .eq(SysAiMemory::getUserId, userId)
                .eq(SysAiMemory::getStatus, 1)
                .eq(SysAiMemory::getArchived, 0)
                .like(SysAiMemory::getContent, keyword)
                .orderByDesc(SysAiMemory::getImportance)
                .last("LIMIT " + limit));
        List<AiMemoryVO> results = memories.stream().map(this::toVO).toList();
        for (SysAiMemory memory : memories) {
            memoryMapper.touch(memory.getId());
        }
        return results;
    }

    @Transactional
    public int batchClear(Long userId, Boolean confirm, String memoryType,
                          LocalDateTime start, LocalDateTime end) {
        if (!Boolean.TRUE.equals(confirm)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "批量清空记忆为不可逆操作，需二次确认");
        }
        int count = memoryMapper.batchClear(userId, memoryType, start, end, userId);
        Map<String, Object> afterValue = new LinkedHashMap<>();
        afterValue.put("count", count);
        afterValue.put("memory_type", memoryType);
        afterValue.put("start", start == null ? null : start.toString());
        afterValue.put("end", end == null ? null : end.toString());
        AiTxSupport.afterCommit(() -> auditLogService.recordAudit(userId, "ai_memory", userId, "clear",
                "ai_memory", null, afterValue, null, null));
        return count;
    }

    @Transactional
    public int restoreDeleted(Long userId, Boolean confirm, String memoryType,
                              LocalDateTime start, LocalDateTime end) {
        if (!Boolean.TRUE.equals(confirm)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "恢复记忆操作需二次确认");
        }
        List<SysAiMemory> deleted = memoryMapper.listDeletedForRestore(userId, memoryType, start, end,
                LocalDateTime.now().minusDays(RECOVERY_WINDOW_DAYS));
        if (deleted.isEmpty()) {
            return 0;
        }
        return memoryMapper.restoreByIds(deleted.stream().map(SysAiMemory::getId).toList(), userId);
    }

    /**
     * 导出用户全部活跃记忆（json/markdown）；记忆属敏感个人数据，批量导出必须留痕
     */
    public MemoryExport export(Long userId, String format) {
        List<SysAiMemory> memories = memoryMapper.selectList(new LambdaQueryWrapper<SysAiMemory>()
                .eq(SysAiMemory::getUserId, userId)
                .eq(SysAiMemory::getStatus, 1)
                .eq(SysAiMemory::getArchived, 0)
                .orderByDesc(SysAiMemory::getImportance)
                .orderByDesc(SysAiMemory::getLastAccessedAt)
                .last("LIMIT 10000"));
        Map<String, Object> afterValue = new LinkedHashMap<>();
        afterValue.put("format", format);
        afterValue.put("count", memories.size());
        auditLogService.recordAudit(userId, "ai_memory", userId, "export", "ai_memory", null, afterValue,
                null, null);
        if ("markdown".equals(format)) {
            StringBuilder sb = new StringBuilder("# 长期记忆导出\n\n");
            for (SysAiMemory memory : memories) {
                sb.append("## ").append(memory.getMemoryType())
                        .append("（来源：").append(memory.getSource()).append("）\n")
                        .append("- 内容：").append(memory.getContent()).append("\n")
                        .append("- 重要性：").append(memory.getImportance()).append("\n")
                        .append("- 创建时间：")
                        .append(memory.getCreateTime() == null ? "-" : memory.getCreateTime().format(TIME_FORMAT))
                        .append("\n\n");
            }
            return new MemoryExport("memories.md", "text/markdown; charset=utf-8", sb.toString());
        }
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("user_id", userId);
        payload.put("exported_at", LocalDateTime.now().toString());
        List<Map<String, Object>> records = new ArrayList<>();
        for (SysAiMemory memory : memories) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", memory.getId());
            item.put("memory_type", memory.getMemoryType());
            item.put("content", memory.getContent());
            item.put("metadata", memory.getMetadata());
            item.put("source", memory.getSource());
            item.put("importance", memory.getImportance());
            item.put("access_count", memory.getAccessCount());
            item.put("created_at",
                    memory.getCreateTime() == null ? null : memory.getCreateTime().format(TIME_FORMAT));
            records.add(item);
        }
        payload.put("memories", records);
        try {
            return new MemoryExport("memories.json", "application/json; charset=utf-8",
                    objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(payload));
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "记忆导出失败");
        }
    }

    private SysAiMemory getOwned(Long memoryId, Long userId) {
        SysAiMemory memory = memoryMapper.selectOne(new LambdaQueryWrapper<SysAiMemory>()
                .eq(SysAiMemory::getId, memoryId)
                .eq(SysAiMemory::getUserId, userId));
        if (memory == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "记忆不存在");
        }
        return memory;
    }

    private AiMemoryVO toVO(SysAiMemory memory) {
        AiMemoryVO vo = new AiMemoryVO();
        vo.setId(memory.getId());
        vo.setUserId(memory.getUserId());
        vo.setMemoryType(memory.getMemoryType());
        vo.setContent(memory.getContent());
        vo.setMetadata(memory.getMetadata());
        vo.setImportance(memory.getImportance());
        vo.setAccessCount(memory.getAccessCount());
        vo.setLastAccessedAt(memory.getLastAccessedAt());
        vo.setSource(memory.getSource());
        vo.setStatus(memory.getStatus());
        vo.setArchived(memory.getArchived());
        vo.setCreateTime(memory.getCreateTime());
        vo.setUpdateTime(memory.getUpdateTime());
        return vo;
    }

    private <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(new ArrayList<>(records));
        return result;
    }
}
