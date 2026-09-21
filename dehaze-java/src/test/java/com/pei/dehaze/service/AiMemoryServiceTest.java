package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiMemoryMapper;
import com.pei.dehaze.model.entity.SysAiMemory;
import com.pei.dehaze.model.form.AiMemoryUpdateForm;
import com.pei.dehaze.service.impl.AiKbIndexClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AI 长期记忆服务单测：归属校验、归档反向操作与衰减计时器刷新、二次确认、检索重激活。
 */
@DisplayName("AiMemoryService 记忆生命周期")
@ExtendWith(MockitoExtension.class)
class AiMemoryServiceTest {

    @Mock
    private SysAiMemoryMapper memoryMapper;

    @Mock
    private AuditLogService auditLogService;

    @Mock
    private AiKbIndexClient kbIndexClient;

    private AiMemoryService service;

    @BeforeEach
    void setUp() {
        service = new AiMemoryService(memoryMapper, auditLogService, new ObjectMapper(), kbIndexClient);
    }

    private SysAiMemory memory(Long id, Long userId, int archived) {
        SysAiMemory memory = new SysAiMemory();
        memory.setId(id);
        memory.setUserId(userId);
        memory.setMemoryType("semantic");
        memory.setContent("用户偏好简洁回复");
        memory.setImportance(50);
        memory.setSource("manual");
        memory.setStatus(1);
        memory.setArchived(archived);
        memory.setAccessCount(1);
        memory.setCreateTime(LocalDateTime.now().minusDays(1));
        return memory;
    }

    @Test
    @DisplayName("取消归档：archived 归零并刷新衰减计时器（否则次日归档任务立即再次归档）")
    void unarchiveResetsArchivedAndDecayTimer() {
        SysAiMemory memory = memory(9L, 3L, 1);
        when(memoryMapper.selectOne(any(Wrapper.class))).thenReturn(memory);

        service.unarchive(9L, 3L);

        assertThat(memory.getArchived()).isZero();
        assertThat(memory.getLastAccessedAt()).isNotNull();
        verify(memoryMapper).updateById(memory);
    }

    @Test
    @DisplayName("未归档记忆取消归档报 A0502")
    void unarchiveRejectsNonArchivedMemory() {
        when(memoryMapper.selectOne(any(Wrapper.class))).thenReturn(memory(9L, 3L, 0));

        assertThatThrownBy(() -> service.unarchive(9L, 3L))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.DATA_STATE_NOT_ALLOW);
        verify(memoryMapper, never()).updateById(any(SysAiMemory.class));
    }

    @Test
    @DisplayName("他人记忆不可见：归属校验失败报 A0401")
    void otherUsersMemoryIsInvisible() {
        when(memoryMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        assertThatThrownBy(() -> service.update(9L, 3L, new AiMemoryUpdateForm()))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
        verify(memoryMapper, never()).updateById(any(SysAiMemory.class));
    }

    @Test
    @DisplayName("批量清空缺二次确认报 A0400，且不触碰存储")
    void batchClearRequiresConfirm() {
        assertThatThrownBy(() -> service.batchClear(3L, null, "semantic", null, null))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verify(memoryMapper, never()).batchClear(any(), any(), any(), any(), any());
    }

    @Test
    @DisplayName("批量清空：筛选条件透传并留审计（count/类型/时间范围）")
    void batchClearPassesFiltersAndAudits() {
        LocalDateTime start = LocalDateTime.now().minusDays(7);
        LocalDateTime end = LocalDateTime.now();
        when(memoryMapper.batchClear(3L, "semantic", start, end, 3L)).thenReturn(4);

        assertThat(service.batchClear(3L, true, "semantic", start, end)).isEqualTo(4);
        verify(auditLogService).recordAudit(eq(3L), eq("ai_memory"), eq(3L), eq("clear"), eq("ai_memory"),
                isNull(), any(), isNull(), isNull());
    }

    @Test
    @DisplayName("恢复软删记忆缺二次确认报 A0400")
    void restoreDeletedRequiresConfirm() {
        assertThatThrownBy(() -> service.restoreDeleted(3L, false, null, null, null))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verify(memoryMapper, never()).restoreByIds(anyList(), any());
    }

    @Test
    @DisplayName("恢复窗口内无软删记忆返回 0 且不发起更新")
    void restoreDeletedReturnsZeroWhenNothingToRestore() {
        when(memoryMapper.listDeletedForRestore(eq(3L), eq(null), eq(null), eq(null), any())).thenReturn(List.of());

        assertThat(service.restoreDeleted(3L, true, null, null, null)).isZero();
        verify(memoryMapper, never()).restoreByIds(anyList(), any());
    }

    @Test
    @DisplayName("恢复窗口为 30 天：窗口起点透传 delete_time 下界")
    void restoreDeletedUsesThirtyDayWindow() {
        when(memoryMapper.listDeletedForRestore(eq(3L), eq("semantic"), eq(null), eq(null), any()))
                .thenReturn(List.of(memory(9L, 3L, 0)));
        when(memoryMapper.restoreByIds(List.of(9L), 3L)).thenReturn(1);

        assertThat(service.restoreDeleted(3L, true, "semantic", null, null)).isEqualTo(1);

        ArgumentCaptor<LocalDateTime> windowCaptor = ArgumentCaptor.forClass(LocalDateTime.class);
        verify(memoryMapper).listDeletedForRestore(eq(3L), eq("semantic"), eq(null), eq(null),
                windowCaptor.capture());
        assertThat(windowCaptor.getValue()).isBefore(LocalDateTime.now().minusDays(29));
    }

    @Test
    @DisplayName("检索命中后逐条重激活（access_count+1 / 刷新衰减 / importance+5）")
    void searchTouchesEveryHit() {
        when(memoryMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(memory(9L, 3L, 0), memory(10L, 3L, 0)));

        assertThat(service.search(3L, "偏好", 5)).hasSize(2);
        verify(memoryMapper).touch(9L);
        verify(memoryMapper).touch(10L);
        verify(memoryMapper, times(2)).touch(any());
    }

    @Test
    @DisplayName("删除记忆：软删后同步清除 ES 向量文档（已删记忆不得被向量检索召回）")
    void deleteClearsEsVectorDoc() {
        when(memoryMapper.selectOne(any(Wrapper.class))).thenReturn(memory(9L, 3L, 0));

        service.delete(9L, 3L);

        verify(memoryMapper).softDeleteByIds(List.of(9L), 3L);
        verify(kbIndexClient).deleteMemoryDoc(9L);
    }
}
