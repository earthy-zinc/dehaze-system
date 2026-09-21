package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiAgentThoughtMapper;
import com.pei.dehaze.mapper.SysAiAgentVersionMapper;
import com.pei.dehaze.mapper.SysAiConversationMapper;
import com.pei.dehaze.mapper.SysAiMessageMapper;
import com.pei.dehaze.model.entity.SysAiAgentThought;
import com.pei.dehaze.model.entity.SysAiConversation;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.vo.AiMessagePageVO;
import com.pei.dehaze.model.vo.AiMessageVO;
import com.pei.dehaze.model.form.AiConversationBatchForm;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.data.redis.core.script.RedisScript;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 会话与消息服务单测：归属校验、置顶名额（用户级锁 + 已置顶不刷新）、批量操作语义、消息删除限制、导出。
 */
@DisplayName("AiConversationService 会话生命周期")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiConversationServiceTest {

    @Mock
    private SysAiConversationMapper conversationMapper;
    @Mock
    private SysAiMessageMapper messageMapper;
    @Mock
    private SysAiAgentThoughtMapper thoughtMapper;
    @Mock
    private SysAiAgentMapper agentMapper;
    @Mock
    private SysAiAgentVersionMapper agentVersionMapper;
    @Mock
    private AiInsightMapper insightMapper;
    @Mock
    private StringRedisTemplate stringRedisTemplate;
    @Mock
    private ValueOperations<String, String> valueOperations;

    @Spy
    private ObjectMapper objectMapper = new ObjectMapper();

    @InjectMocks
    private AiConversationService service;

    private static final Long USER_ID = 3L;

    /**
     * 纯单测无 MP 启动流程，需手动初始化 SysAiMessage 的 lambda 缓存，
     * 否则游标用例中渲染 LambdaQueryWrapper 会抛 "can not find lambda cache"。
     */
    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "cursor-test");
        assistant.setCurrentNamespace("com.pei.dehaze.mapper.SysAiMessageMapper");
        TableInfoHelper.initTableInfo(assistant, SysAiMessage.class);
    }

    @BeforeEach
    void setUp() {
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        // 消息表无 user_id，归属由所属会话判定：会话存在即视为本人消息
        when(conversationMapper.selectCount(any(Wrapper.class))).thenReturn(1L);
    }

    private SysAiConversation conv(Long id, Long userId, Integer pinned) {
        SysAiConversation conv = new SysAiConversation();
        conv.setId(id);
        conv.setUserId(userId);
        conv.setTitle("会话");
        conv.setStatus(1);
        conv.setPinned(pinned == null ? 0 : pinned);
        conv.setMessageCount(2);
        conv.setCreateTime(LocalDateTime.now().minusHours(1));
        return conv;
    }

    private void stubOwned(SysAiConversation conv) {
        when(conversationMapper.selectOne(any(Wrapper.class))).thenReturn(conv);
    }

    @Test
    @DisplayName("会话归属校验：非本人或不存在一律 A0401")
    void nonOwnedConversationIsNotFound() {
        when(conversationMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.getOwned(1L, USER_ID));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
    }

    @Test
    @DisplayName("删除消息：仅助手消息可删，用户消息报 A0502")
    void deleteUserMessageRejected() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage message = new SysAiMessage();
        message.setId(9L);
        message.setConversationId(1L);
        message.setRole("user");
        when(messageMapper.selectById(9L)).thenReturn(message);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.deleteMessage(9L, USER_ID));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.DATA_STATE_NOT_ALLOW);
        verify(messageMapper, never()).softDeleteByIds(anyList());
    }

    @Test
    @DisplayName("删除消息：助手消息软删")
    void deleteAssistantMessageSoftDeletes() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage message = new SysAiMessage();
        message.setId(9L);
        message.setConversationId(1L);
        message.setRole("assistant");
        when(messageMapper.selectById(9L)).thenReturn(message);

        service.deleteMessage(9L, USER_ID);

        verify(messageMapper).softDeleteByIds(List.of(9L));
    }

    @Test
    @DisplayName("批量删除缺二次确认报 A0400，且不触碰存储")
    void batchDeleteRequiresConfirm() {
        stubOwned(conv(1L, USER_ID, 0));
        AiConversationBatchForm form = new AiConversationBatchForm();
        form.setAction("delete");
        form.setIds(List.of(1L));

        BusinessException ex = assertThrows(BusinessException.class, () -> service.batchOperate(USER_ID, form));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verify(conversationMapper, never()).softDeleteByIds(anyList());
    }

    @Test
    @DisplayName("批量归档：非活跃会话报 A0502")
    void batchArchiveRejectsArchivedConversation() {
        SysAiConversation archived = conv(1L, USER_ID, 0);
        archived.setStatus(2);
        stubOwned(archived);
        AiConversationBatchForm form = new AiConversationBatchForm();
        form.setAction("archive");
        form.setIds(List.of(1L));

        BusinessException ex = assertThrows(BusinessException.class, () -> service.batchOperate(USER_ID, form));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.DATA_STATE_NOT_ALLOW);
        verify(conversationMapper, never()).updateStatusByIds(anyList(), any());
    }

    @Test
    @DisplayName("批量归档/恢复：逐条复用单条校验并返回处理条数")
    void batchArchiveAndRestoreUpdateStatus() {
        stubOwned(conv(1L, USER_ID, 0));
        AiConversationBatchForm form = new AiConversationBatchForm();
        form.setAction("archive");
        form.setIds(List.of(1L));

        assertThat(service.batchOperate(USER_ID, form)).isEqualTo(1);
        verify(conversationMapper).updateStatusByIds(List.of(1L), 2);
    }

    @Test
    @DisplayName("批量操作未知动作报 A0400")
    void batchUnknownActionRejected() {
        stubOwned(conv(1L, USER_ID, 0));
        AiConversationBatchForm form = new AiConversationBatchForm();
        form.setAction("nuke");
        form.setIds(List.of(1L));

        BusinessException ex = assertThrows(BusinessException.class, () -> service.batchOperate(USER_ID, form));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
    }

    @Test
    @DisplayName("置顶：已置顶会话不再刷新 pinned_at（避免打乱置顶排序）")
    void pinAlreadyPinnedKeepsPinnedAt() {
        stubOwned(conv(1L, USER_ID, 1));

        service.pin(1L, USER_ID);

        verify(conversationMapper, never()).setPinned(any(), any(), any());
        verify(conversationMapper, never()).countActivePinned(any());
    }

    @Test
    @DisplayName("置顶：超过名额上限报 A0501")
    void pinRejectsWhenLimitReached() {
        stubOwned(conv(1L, USER_ID, 0));
        when(valueOperations.setIfAbsent(anyString(), anyString(), any())).thenReturn(true);
        when(conversationMapper.countActivePinned(USER_ID)).thenReturn(10L);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.pin(1L, USER_ID));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.DATA_EXISTS);
        verify(conversationMapper, never()).setPinned(any(), any(), any());
    }

    @Test
    @DisplayName("置顶：名额校验与写入在用户级锁内完成，锁在 finally 释放")
    void pinHoldsUserLevelLockAndReleasesIt() {
        stubOwned(conv(1L, USER_ID, 0));
        when(valueOperations.setIfAbsent(eq("ai:conv:pin:" + USER_ID), anyString(), any())).thenReturn(true);
        when(conversationMapper.countActivePinned(USER_ID)).thenReturn(0L);

        service.pin(1L, USER_ID);

        verify(conversationMapper).setPinned(eq(1L), eq(1), any(LocalDateTime.class));
        verify(stringRedisTemplate).execute(any(RedisScript.class), eq(List.of("ai:conv:pin:" + USER_ID)), any());
    }

    @Test
    @DisplayName("置顶：并发抢不到用户级锁报 A0500")
    void pinReportsConflictWhenLockUnavailable() {
        stubOwned(conv(1L, USER_ID, 0));
        when(valueOperations.setIfAbsent(anyString(), anyString(), any())).thenReturn(false);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.pin(1L, USER_ID));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.BUSINESS_ERROR);
    }

    @Test
    @DisplayName("取消置顶：清空 pinned 与 pinned_at")
    void unpinClearsPinnedTimestamp() {
        stubOwned(conv(1L, USER_ID, 1));
        when(messageMapper.countMessagesAfter(any(), any())).thenReturn(0L);

        service.unpin(1L, USER_ID);

        verify(conversationMapper).setPinned(1L, 0, null);
    }

    @Test
    @DisplayName("已读：标记到最新消息（无消息时不写入）")
    void markReadUsesLastMessageId() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.getLastMessageId(1L)).thenReturn(7L);
        when(messageMapper.countMessagesAfter(1L, 7L)).thenReturn(0L);

        assertThat(service.markRead(1L, USER_ID).getLastReadMessageId()).isEqualTo(7L);
        verify(conversationMapper).markRead(1L, 7L);
    }

    @Test
    @DisplayName("会话消息列表：普通用户校验归属（不存在即 A0401）")
    void listMessagesEnforcesOwnershipForNormalUser() {
        when(conversationMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.listMessages(1L, USER_ID, null, 50, false));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
    }

    @Test
    @DisplayName("消息列表：默认取最新一页，末页 hasMore=false")
    void listMessagesDefaultPageHasNoMore() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(messagePage(List.of(message(5L, "user", "e"), message(4L, "user", "d"))));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(2L);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 2, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(2L);
        assertThat(result.isHasMore()).isFalse();
    }

    @Test
    @DisplayName("消息列表：hasMore=true 时按 limit+1 单次取数并裁掉多余一条（不额外往返）")
    void listMessagesHasMoreTrimsExtraRow() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class))).thenReturn(messagePage(
                List.of(message(5L, "user", "5"), message(4L, "user", "4"), message(3L, "user", "3"))));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(9L);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 2, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(9L);
        assertThat(result.isHasMore()).isTrue();

        ArgumentCaptor<Page<SysAiMessage>> captor = ArgumentCaptor.forClass(Page.class);
        verify(messageMapper).selectPage(captor.capture(), any(Wrapper.class));
        assertThat(captor.getValue().getSize()).isEqualTo(3L);
        assertThat(captor.getValue().searchCount()).isFalse();
    }

    @Test
    @DisplayName("消息列表：空会话返回空列表 + total=0 + hasMore=false")
    void listMessagesEmptyConversation() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class))).thenReturn(messagePage(List.of()));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).isEmpty();
        assertThat(result.getTotal()).isZero();
        assertThat(result.isHasMore()).isFalse();
    }

    @Test
    @DisplayName("消息列表：assistant 消息仍附带推理步骤（组装逻辑保持不变）")
    void listMessagesAttachesThoughts() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(messagePage(List.of(message(5L, "assistant", "reply"))));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(1L);
        SysAiAgentThought thought = new SysAiAgentThought();
        thought.setMessageId(5L);
        thought.setCreateTime(LocalDateTime.now());
        when(thoughtMapper.selectList(any(Wrapper.class))).thenReturn(List.of(thought));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList().get(0).getThoughts()).hasSize(1);
    }

    @Test
    @DisplayName("消息列表（admin 视角）：不做归属过滤，被审计会话仍可访问")
    void listMessagesAdminBypassesOwnership() {
        SysAiConversation foreign = conv(1L, 999L, 0);
        when(conversationMapper.selectById(1L)).thenReturn(foreign);
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(messagePage(List.of(message(5L, "user", "x"))));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(1L);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, true);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L);
    }

    @Test
    @DisplayName("消息列表：并发插入下按 before 继续翻页不重不漏（新消息只出现在最新页）")
    void listMessagesCursorStableUnderConcurrentInsert() {
        stubOwned(conv(1L, USER_ID, 0));
        List<SysAiMessage> store = new ArrayList<>();
        for (long id = 1; id <= 5; id++) {
            store.add(message(id, "user", "m" + id));
        }
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class))).thenAnswer(inv -> {
            Page<SysAiMessage> requested = inv.getArgument(0);
            Long before = cursorBefore(inv.getArgument(1));
            int size = (int) requested.getSize();
            List<SysAiMessage> slice = store.stream()
                    .filter(m -> before == null || m.getId() < before)
                    .sorted(Comparator.comparing(SysAiMessage::getId).reversed())
                    .limit(size)
                    .toList();
            return messagePage(slice);
        });
        when(messageMapper.selectCount(any(Wrapper.class))).thenAnswer(inv -> (long) store.size());

        AiMessagePageVO p1 = service.listMessages(1L, USER_ID, null, 2, false);
        assertThat(p1.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(p1.isHasMore()).isTrue();

        // 翻页途中并发插入更新的消息（id=6，属"最新页"，不应污染后续游标页）
        store.add(message(6L, "user", "m6"));

        Long cursor = p1.getList().get(p1.getList().size() - 1).getId();
        AiMessagePageVO p2 = service.listMessages(1L, USER_ID, cursor, 2, false);
        assertThat(p2.getList()).extracting(AiMessageVO::getId).containsExactly(3L, 2L);
        assertThat(p2.isHasMore()).isTrue();

        AiMessagePageVO p3 = service.listMessages(1L, USER_ID, p2.getList().get(1).getId(), 2, false);
        assertThat(p3.getList()).extracting(AiMessageVO::getId).containsExactly(1L);
        assertThat(p3.isHasMore()).isFalse();

        List<Long> merged = new ArrayList<>();
        merged.addAll(p1.getList().stream().map(AiMessageVO::getId).toList());
        merged.addAll(p2.getList().stream().map(AiMessageVO::getId).toList());
        merged.addAll(p3.getList().stream().map(AiMessageVO::getId).toList());
        assertThat(merged).containsExactly(5L, 4L, 3L, 2L, 1L).doesNotContain(6L);
        // total 为"该会话消息总数"，随并发插入增长
        assertThat(p3.getTotal()).isEqualTo(6L);
    }

    @Test
    @DisplayName("消息列表：before 超出 Integer.MAX_VALUE 时按完整 long 参与游标查询（不被截断）")
    void listMessagesBeforeAboveIntegerMaxNotTruncated() {
        stubOwned(conv(1L, USER_ID, 0));
        long bigBefore = (long) Integer.MAX_VALUE + 10L;
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(messagePage(List.of(message(5L, "user", "x"))));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(1L);

        service.listMessages(1L, USER_ID, bigBefore, 1, false);

        ArgumentCaptor<Wrapper<SysAiMessage>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(messageMapper).selectPage(any(Page.class), captor.capture());
        assertThat(cursorBefore(captor.getValue())).isEqualTo(bigBefore);
    }

    private static Page<SysAiMessage> messagePage(List<SysAiMessage> records) {
        Page<SysAiMessage> page = new Page<>(1, records.size());
        page.setRecords(new ArrayList<>(records));
        return page;
    }

    /** 取游标 before：MP 参数按 MPGENVAL 序号登记（1=conversationId，2=before），渲染后才填充 */
    private static Long cursorBefore(Wrapper<SysAiMessage> wrapper) {
        AbstractWrapper<?, ?, ?> inner = (AbstractWrapper<?, ?, ?>) wrapper;
        inner.getSqlSegment();
        return inner.getParamNameValuePairs().entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(entry -> ((Number) entry.getValue()).longValue())
                .skip(1)
                .findFirst()
                .orElse(null);
    }

    @Test
    @DisplayName("导出 markdown：仅含 user/assistant 正文，跳过推理与工具消息")
    void exportMarkdownSkipsThoughtMessages() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(3L);
        stubOwned(conv);
        SysAiMessage tool = message(2L, "tool", "{\"tool\":\"x\"}");
        tool.setParentMessageId(1L);
        SysAiMessage assistant = message(3L, "assistant", "已处理");
        assistant.setParentMessageId(2L);
        when(messageMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(message(1L, "user", "帮我处理图片"), tool, assistant));

        AiConversationService.ConversationExport export = service.export(1L, USER_ID, "markdown");

        assertThat(export.contentType()).isEqualTo("text/markdown");
        assertThat(export.content())
                .contains("帮我处理图片")
                .contains("已处理")
                .doesNotContain("tool");
    }

    @Test
    @DisplayName("导出 json：消息体仅含 role/content/create_time")
    void exportJsonContainsConversationAndMessages() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(2L);
        stubOwned(conv);
        when(messageMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(message(1L, "user", "问题"), message(2L, "assistant", "回答")));

        AiConversationService.ConversationExport export =
                service.export(1L, USER_ID, "json");

        assertThat(export.contentType()).isEqualTo("application/json");
        assertThat(export.content())
                .contains("\"conversation\"")
                .contains("\"agent_code\"")
                .contains("\"messages\"");
    }

    @Test
    @DisplayName("回收站恢复：超出 30 天窗口报 A0401")
    void restoreOutsideWindowRejected() {
        when(conversationMapper.selectInTrash(eq(1L), eq(USER_ID), any(LocalDateTime.class))).thenReturn(null);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.restore(1L, USER_ID));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
        assertThat(ex.getMessage()).contains("恢复窗口");
    }

    @Test
    @DisplayName("回收站列表：窗口起点为 30 天前")
    void trashListUsesThirtyDayWindow() {
        when(conversationMapper.selectTrashPage(any(Page.class), eq(USER_ID), any(LocalDateTime.class)))
                .thenReturn(new Page<>(1, 10, 0));

        service.listTrash(USER_ID, 1, 10);

        org.mockito.ArgumentCaptor<LocalDateTime> captor = org.mockito.ArgumentCaptor.forClass(LocalDateTime.class);
        verify(conversationMapper).selectTrashPage(any(Page.class), eq(USER_ID), captor.capture());
        assertThat(captor.getValue()).isBefore(LocalDateTime.now().minusDays(29));
    }

    @Test
    @DisplayName("删除会话：软删后 30 天内可从回收站恢复")
    void deleteSoftDeletesConversation() {
        stubOwned(conv(1L, USER_ID, 0));

        service.delete(1L, USER_ID);

        verify(conversationMapper).softDeleteByIds(List.of(1L));
    }

    @Test
    @DisplayName("会话详情（普通用户）：与 python 一致不做未读换算（不额外查库）")
    void detailForNormalUserSkipsUnreadComputation() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setLastReadMessageId(5L);
        stubOwned(conv);

        assertThat(service.getDetail(1L, USER_ID, false).getUnreadCount()).isZero();
        verify(messageMapper, never()).countMessagesAfter(any(), any());
    }

    @Test
    @DisplayName("会话详情（管理端）：按已读指针换算未读数")
    void adminDetailComputesUnreadCount() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setLastReadMessageId(5L);
        when(conversationMapper.selectById(1L)).thenReturn(conv);
        when(messageMapper.countMessagesAfter(1L, 5L)).thenReturn(3L);
        when(insightMapper.listUserDisplayNames(anyList())).thenReturn(List.of());
        when(insightMapper.sumConsumptionByConversationIds(anyList())).thenReturn(List.of());
        when(insightMapper.listAnomalyStatusByConversations(anyList())).thenReturn(List.of());
        when(insightMapper.listQuotaAnomalyConversationIds(anyList())).thenReturn(List.of());
        when(insightMapper.listRiskyToolConversationIds(anyList())).thenReturn(List.of());

        assertThat(service.getDetail(1L, USER_ID, true).getUnreadCount()).isEqualTo(3);
    }

    private SysAiMessage message(Long id, String role, String content) {
        SysAiMessage message = new SysAiMessage();
        message.setId(id);
        message.setConversationId(1L);
        message.setRole(role);
        message.setContent(content);
        message.setCreateTime(LocalDateTime.now());
        return message;
    }

    @Test
    @DisplayName("会话详情（管理端）：不限归属，被审计会话仍可访问")
    void adminDetailBypassesOwnership() {
        SysAiConversation conv = conv(1L, 999L, 0);
        conv.setUserId(999L);
        when(conversationMapper.selectById(1L)).thenReturn(conv);
        when(messageMapper.countMessagesAfter(any(), any())).thenReturn(0L);
        when(insightMapper.listUserDisplayNames(anyList())).thenReturn(List.of());
        when(insightMapper.sumConsumptionByConversationIds(anyList())).thenReturn(List.of());
        when(insightMapper.listAnomalyStatusByConversations(anyList())).thenReturn(List.of());
        when(insightMapper.listQuotaAnomalyConversationIds(anyList())).thenReturn(List.of());
        when(insightMapper.listRiskyToolConversationIds(anyList())).thenReturn(List.of());

        assertThat(service.getDetail(1L, USER_ID, true).getUserId()).isEqualTo(999L);
    }

    @Test
    @DisplayName("会话详情（管理端）：会话不存在报 A0401")
    void adminDetailRejectsMissingConversation() {
        when(conversationMapper.selectById(1L)).thenReturn(null);

        assertThatThrownBy(() -> service.getDetail(1L, USER_ID, true))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
    }
}
