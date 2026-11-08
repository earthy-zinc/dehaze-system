package com.pei.clarity.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.AiInsightMapper;
import com.pei.clarity.mapper.SysAiAgentMapper;
import com.pei.clarity.mapper.SysAiAgentVersionMapper;
import com.pei.clarity.mapper.SysAiArtifactMapper;
import com.pei.clarity.mapper.SysAiConversationMapper;
import com.pei.clarity.mapper.SysAiMessageMapper;
import com.pei.clarity.model.dto.ai.AiContract;
import com.pei.clarity.model.entity.SysAiConversation;
import com.pei.clarity.model.entity.SysAiMessage;
import com.pei.clarity.model.form.AiConversationBatchForm;
import com.pei.clarity.model.form.AiConversationCreateForm;
import com.pei.clarity.model.query.AiConversationPageQuery;
import com.pei.clarity.model.vo.AiConversationVO;
import com.pei.clarity.model.vo.AiMessagePageVO;
import com.pei.clarity.model.vo.AiMessageVO;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.data.redis.core.script.RedisScript;

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
    private SysAiArtifactMapper artifactMapper;

    @Mock
    private SysAiAgentMapper agentMapper;

    @Mock
    private SysAiAgentVersionMapper agentVersionMapper;

    @Mock
    private AiInsightMapper insightMapper;

    @Mock
    private AiDefaultModelResolver defaultModelResolver;

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
     * 纯单测无 MP 启动流程，需手动初始化实体 lambda 缓存，
     * 否则渲染 LambdaQueryWrapper 会抛 "can not find lambda cache"。
     */
    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant messageAssistant = new MapperBuilderAssistant(configuration, "cursor-test");
        messageAssistant.setCurrentNamespace("com.pei.clarity.mapper.SysAiMessageMapper");
        TableInfoHelper.initTableInfo(messageAssistant, SysAiMessage.class);
        MapperBuilderAssistant conversationAssistant = new MapperBuilderAssistant(configuration, "conversation-test");
        conversationAssistant.setCurrentNamespace("com.pei.clarity.mapper.SysAiConversationMapper");
        TableInfoHelper.initTableInfo(conversationAssistant, SysAiConversation.class);
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
        // deleteMessage 事务内先以 FOR UPDATE 锁会话行，再做 children 检查与指针回退
        when(conversationMapper.selectByIdForUpdate(conv.getId(), conv.getUserId()))
                .thenReturn(conv);
    }

    @Test
    @DisplayName("创建会话未指定模型：走解析链取用户偏好/平台默认")
    void createWithoutModelUsesResolvedDefault() {
        when(defaultModelResolver.resolveForUser(USER_ID, "default")).thenReturn("qwen3-0.6b");
        AiConversationCreateForm form = new AiConversationCreateForm();

        AiConversationVO vo = service.create(USER_ID, form);

        assertThat(vo.getModel()).isEqualTo("qwen3-0.6b");
        verify(defaultModelResolver).resolveForUser(USER_ID, "default");
        verify(conversationMapper).insert(any(SysAiConversation.class));
    }

    @Test
    @DisplayName("创建会话显式指定模型：不触达解析链")
    void createWithExplicitModelSkipsResolver() {
        AiConversationCreateForm form = new AiConversationCreateForm();
        form.setModel("gpt-4o");

        assertThat(service.create(USER_ID, form).getModel()).isEqualTo("gpt-4o");
        verify(defaultModelResolver, never()).resolveForUser(any(), any());
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
    @DisplayName("删除消息：删除当前分支尾消息时指针回退到父消息，历史链仍可读")
    void deleteTailMessageRepointsBranchPointerToParent() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        SysAiMessage message = message(9L, "assistant", "tail");
        message.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(message);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(9L, USER_ID);

        verify(messageMapper).softDeleteByIds(List.of(9L));
        verify(conversationMapper).compareAndSetCurrentBranch(1L, 9L, 8L);
    }

    @Test
    @DisplayName("删除消息：尾部无父消息（root 级）时指针清空为 null")
    void deleteRootTailMessageClearsBranchPointer() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        SysAiMessage message = message(9L, "assistant", "root");
        when(messageMapper.selectById(9L)).thenReturn(message);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(9L, USER_ID);

        verify(conversationMapper).compareAndSetCurrentBranch(1L, 9L, null);
    }

    @Test
    @DisplayName("删除消息：仍有未删子消息（非叶子）拒绝删除，消息保留、不改指针")
    void deleteMessageWithUndeletedChildrenRejected() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        SysAiMessage message = message(9L, "assistant", "middle");
        message.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(message);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(1L);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.deleteMessage(9L, USER_ID));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.DATA_STATE_NOT_ALLOW);
        assertThat(ex.getMessage()).contains("请先删除该回复之后的对话");
        verify(messageMapper, never()).softDeleteByIds(anyList());
        verify(conversationMapper, never()).compareAndSetCurrentBranch(anyLong(), anyLong(), any());
        // 删除校验看全部子消息（含 user 子消息），不得加 role 过滤
        ArgumentCaptor<Wrapper<SysAiMessage>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(messageMapper).selectCount(captor.capture());
        assertThat(captor.getValue().getSqlSegment()).doesNotContain("role");
    }

    @Test
    @DisplayName("删除消息：非激活分支上的旧分支消息允许删除，指针不动")
    void deleteInactiveBranchMessageKeepsPointer() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        SysAiMessage oldBranch = message(5L, "assistant", "old");
        oldBranch.setParentMessageId(4L);
        when(messageMapper.selectById(5L)).thenReturn(oldBranch);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(5L, USER_ID);

        verify(messageMapper).softDeleteByIds(List.of(5L));
        verify(conversationMapper, never()).compareAndSetCurrentBranch(anyLong(), anyLong(), any());
    }

    @Test
    @DisplayName("删除消息：删尾部后继续发送，指针从父消息续接，历史消息不丢")
    void deleteTailThenContinueSendingKeepsHistory() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(4L);
        stubOwned(conv);
        SysAiMessage tail = message(4L, "assistant", "tail");
        tail.setParentMessageId(3L);
        when(messageMapper.selectById(4L)).thenReturn(tail);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(4L, USER_ID);
        verify(conversationMapper).compareAndSetCurrentBranch(1L, 4L, 3L);

        // 继续发送：新 user 消息挂在被删消息的父消息上，指针推进到新消息
        SysAiMessage user1 = message(1L, "user", "ask");
        SysAiMessage reply1 = message(2L, "assistant", "first");
        reply1.setParentMessageId(1L);
        SysAiMessage user2 = message(3L, "user", "follow-up");
        user2.setParentMessageId(2L);
        SysAiMessage next = message(5L, "user", "next");
        next.setParentMessageId(3L);
        conv.setCurrentBranchMessageId(5L);
        stubActiveChain(conv, List.of(user1, reply1, user2, next));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 3L, 2L, 1L);
        assertThat(result.getTotal()).isEqualTo(4L);
    }

    @Test
    @DisplayName("删除消息：事务内先锁会话行，再做 children 检查、软删与指针回退")
    void deleteMessageLocksConversationBeforeChildCheck() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        SysAiMessage tail = message(9L, "assistant", "tail");
        tail.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(tail);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(9L, USER_ID);

        InOrder order = inOrder(conversationMapper, messageMapper);
        order.verify(conversationMapper).selectByIdForUpdate(1L, USER_ID);
        order.verify(messageMapper).selectCount(any(Wrapper.class));
        order.verify(messageMapper).softDeleteByIds(List.of(9L));
        order.verify(conversationMapper).compareAndSetCurrentBranch(1L, 9L, 8L);
    }

    @Test
    @DisplayName("删除消息：send 已推进指针后删旧尾，不回退也不覆盖新指针")
    void deleteOldTailAfterSendAdvancedPointerKeepsNewPointer() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(10L);
        stubOwned(conv);
        SysAiMessage oldTail = message(9L, "assistant", "old-tail");
        oldTail.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(oldTail);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);

        service.deleteMessage(9L, USER_ID);

        verify(messageMapper).softDeleteByIds(List.of(9L));
        verify(conversationMapper, never()).compareAndSetCurrentBranch(anyLong(), anyLong(), any());
        verify(conversationMapper, never()).updateCurrentBranch(anyLong(), any());
    }

    @Test
    @DisplayName("删除消息：指针回写走 CAS，窗口内被并发推进时保留新指针")
    void deleteMessageCasSkipsWriteWhenPointerAdvancedConcurrently() {
        SysAiConversation snapshot = conv(1L, USER_ID, 0);
        snapshot.setCurrentBranchMessageId(9L);
        stubOwned(snapshot);
        SysAiMessage tail = message(9L, "assistant", "tail");
        tail.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(tail);
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(0L);
        // CAS 未命中：库内指针已被 send 推进到 10，此处返回 0 表示不覆盖
        when(conversationMapper.compareAndSetCurrentBranch(1L, 9L, 8L)).thenReturn(0);

        service.deleteMessage(9L, USER_ID);

        verify(conversationMapper).compareAndSetCurrentBranch(1L, 9L, 8L);
        verify(conversationMapper, never()).updateCurrentBranch(anyLong(), any());
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
        when(valueOperations.setIfAbsent(eq("ai:conv:pin:" + USER_ID), anyString(), any()))
                .thenReturn(true);
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

        BusinessException ex =
                assertThrows(BusinessException.class, () -> service.listMessages(1L, USER_ID, null, 50, false));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
    }

    @Test
    @DisplayName("消息列表：默认取激活链最新一页，末页 hasMore=false，total 为链长")
    void listMessagesDefaultPageHasNoMore() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        SysAiMessage user = message(4L, "user", "d");
        SysAiMessage assistant = message(5L, "assistant", "e");
        assistant.setParentMessageId(4L);
        stubActiveChain(conv, List.of(user, assistant));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 2, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(2L);
        assertThat(result.isHasMore()).isFalse();
    }

    @Test
    @DisplayName("消息列表：分页在链上切，hasMore 表示链上还有更早消息（不做全表游标查询）")
    void listMessagesHasMoreTrimsToLatestPage() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        SysAiMessage m3 = message(3L, "user", "3");
        SysAiMessage m4 = message(4L, "user", "4");
        m4.setParentMessageId(3L);
        SysAiMessage m5 = message(5L, "user", "5");
        m5.setParentMessageId(4L);
        stubActiveChain(conv, List.of(m3, m4, m5));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 2, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(3L);
        assertThat(result.isHasMore()).isTrue();
        verify(messageMapper, never()).selectPage(any(Page.class), any(Wrapper.class));
    }

    @Test
    @DisplayName("消息列表：空会话返回空列表 + total=0 + hasMore=false")
    void listMessagesEmptyConversation() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        stubOwned(conv);
        when(messageMapper.getMaxMessageId(1L)).thenReturn(null);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).isEmpty();
        assertThat(result.getTotal()).isZero();
        assertThat(result.isHasMore()).isFalse();
    }

    @Test
    @DisplayName("消息列表：指针为空时回退到会话最大 id 未删消息为链尾")
    void listMessagesFallsBackToMaxIdWhenPointerMissing() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        SysAiMessage m4 = message(4L, "user", "4");
        m4.setParentMessageId(3L);
        SysAiMessage m5 = message(5L, "assistant", "5");
        m5.setParentMessageId(4L);
        stubActiveChain(conv, List.of(m4, m5));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(2L);
        verify(messageMapper).getMaxMessageId(1L);
    }

    @Test
    @DisplayName("消息列表：指针悬空（消息不存在）或指向其他会话时回退到最大 id")
    void listMessagesFallsBackWhenPointerDanglingOrForeign() {
        SysAiConversation dangling = conv(1L, USER_ID, 0);
        dangling.setCurrentBranchMessageId(99L);
        SysAiMessage only = message(5L, "user", "x");
        stubActiveChain(dangling, List.of(only));
        when(messageMapper.selectById(99L)).thenReturn(null);
        when(messageMapper.getMaxMessageId(1L)).thenReturn(5L);

        assertThat(service.listMessages(1L, USER_ID, null, 50, false).getList())
                .extracting(AiMessageVO::getId)
                .containsExactly(5L);

        SysAiConversation foreign = conv(1L, USER_ID, 0);
        foreign.setCurrentBranchMessageId(7L);
        SysAiMessage notInConv = message(7L, "assistant", "other");
        notInConv.setConversationId(2L);
        stubOwned(foreign);
        when(messageMapper.selectById(7L)).thenReturn(notInConv);
        when(messageMapper.getMaxMessageId(1L)).thenReturn(5L);
        when(messageMapper.selectList(any(Wrapper.class))).thenReturn(List.of(only));

        assertThat(service.listMessages(1L, USER_ID, null, 50, false).getList())
                .extracting(AiMessageVO::getId)
                .containsExactly(5L);
    }

    @Test
    @DisplayName("消息列表：中间节点软删时链在该处断开，只返回断点之后的活跃后缀")
    void listMessagesTruncatesChainAtSoftDeletedNode() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        // 1-2-3-4-5 中 3 已软删（selectList 不再返回），回溯到 4 时父节点缺失即断链
        SysAiMessage m4 = message(4L, "user", "4");
        m4.setParentMessageId(3L);
        SysAiMessage m5 = message(5L, "assistant", "5");
        m5.setParentMessageId(4L);
        stubActiveChain(conv, List.of(m4, m5));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(result.getTotal()).isEqualTo(2L);
    }

    @Test
    @DisplayName("消息列表：分支消息不混入激活链（total 为链长而非会话消息数）")
    void listMessagesOnlyReturnsActiveChain() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(2L);
        SysAiMessage user = message(1L, "user", "ask");
        SysAiMessage active = message(2L, "assistant", "active");
        active.setParentMessageId(1L);
        SysAiMessage otherBranch = message(3L, "assistant", "branch");
        otherBranch.setParentMessageId(1L);
        stubActiveChain(conv, List.of(user, active, otherBranch));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(2L, 1L);
        assertThat(result.getTotal()).isEqualTo(2L);
    }

    @Test
    @DisplayName("消息列表：assistant 返回兄弟序号（1 起）与兄弟总数，user 消息保持 null")
    void listMessagesFillsSiblingPositionForAssistant() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(3L);
        SysAiMessage user = message(1L, "user", "ask");
        SysAiMessage first = message(2L, "assistant", "first");
        first.setParentMessageId(1L);
        SysAiMessage active = message(3L, "assistant", "second");
        active.setParentMessageId(1L);
        stubActiveChain(conv, List.of(user, first, active));
        when(messageMapper.listAssistantSiblingIds(1L, List.of(1L)))
                .thenReturn(List.of(siblingRow(1L, 2L), siblingRow(1L, 3L)));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        AiMessageVO activeVo = result.getList().stream()
                .filter(vo -> vo.getId().equals(3L))
                .findFirst()
                .orElseThrow();
        assertThat(activeVo.getSiblingIndex()).isEqualTo(2);
        assertThat(activeVo.getSiblingCount()).isEqualTo(2);
        AiMessageVO userVo = result.getList().stream()
                .filter(vo -> vo.getId().equals(1L))
                .findFirst()
                .orElseThrow();
        assertThat(userVo.getSiblingIndex()).isNull();
        assertThat(userVo.getSiblingCount()).isNull();
        // 一次 IN 查询覆盖整页兄弟分组（同一 parent 只查一次）
        verify(messageMapper).listAssistantSiblingIds(1L, List.of(1L));
    }

    @Test
    @DisplayName("消息列表：root 级 assistant（无父消息）不参与兄弟分组，字段保持 null")
    void listMessagesLeavesSiblingNullForRootAssistant() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(6L);
        SysAiMessage rootAssistant = message(6L, "assistant", "root");
        stubActiveChain(conv, List.of(rootAssistant));

        AiMessageVO vo =
                service.listMessages(1L, USER_ID, null, 50, false).getList().get(0);

        assertThat(vo.getSiblingIndex()).isNull();
        assertThat(vo.getSiblingCount()).isNull();
        verify(messageMapper, never()).listAssistantSiblingIds(any(), anyList());
    }

    @Test
    @DisplayName("消息列表：assistant 消息装配产物引用（一次批量查询覆盖整页）")
    void listMessagesAttachesArtifactIds() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        SysAiMessage ask = message(4L, "user", "ask");
        SysAiMessage reply = message(5L, "assistant", "reply");
        reply.setParentMessageId(4L);
        stubActiveChain(conv, List.of(ask, reply));
        when(artifactMapper.listRefsByMessageIds(anyList()))
                .thenReturn(List.of(artifactRef(5L, 101L), artifactRef(5L, 102L)));

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, false);

        assertThat(result.getList().get(0).getArtifactIds()).containsExactly(101L, 102L);
        assertThat(result.getList().get(1).getArtifactIds()).isEmpty();
        // 一次批量查询覆盖整页消息，不逐条回表
        verify(artifactMapper).listRefsByMessageIds(List.of(5L, 4L));
    }

    @Test
    @DisplayName("消息列表：status 按契约字符串透出，正文在 blocks 中（无 content 投影）")
    void listMessagesExposesContractStatusAndBlocks() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        SysAiMessage awaiting = message(5L, "assistant", "请确认");
        awaiting.setStatus("awaiting_input");
        awaiting.setPlan(plan());
        stubActiveChain(conv, List.of(awaiting));
        when(artifactMapper.listRefsByMessageIds(anyList())).thenReturn(List.of());

        AiMessageVO vo =
                service.listMessages(1L, USER_ID, null, 50, false).getList().get(0);

        assertThat(vo.getStatus()).isEqualTo("awaiting_input");
        assertThat(vo.getBlocks()).extracting(AiContract.Block::kind).containsExactly("text");
        assertThat(vo.getBlocks().get(0).text()).isEqualTo("请确认");
        assertThat(vo.getPlan().status()).isEqualTo("running");
        assertThat(vo.getUsage().inputTokens()).isEqualTo(12L);
        assertThat(vo.isEdited()).isTrue();
        assertThat(vo.getMemoryIds()).containsExactly(7L, 8L);
        assertThat(vo.getSuggestions()).containsExactly("继续", "换一个");
    }

    @Test
    @DisplayName("消息详情：单条消息同样装配 artifactIds 与契约字段")
    void getMessageAssemblesArtifactIds() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage msg = message(9L, "assistant", "回答");
        msg.setStatus("completed");
        when(messageMapper.selectById(9L)).thenReturn(msg);
        when(artifactMapper.listRefsByMessageIds(anyList())).thenReturn(List.of(artifactRef(9L, 33L)));

        AiMessageVO vo = service.getMessage(9L, USER_ID, false);

        assertThat(vo.getArtifactIds()).containsExactly(33L);
        assertThat(vo.getStatus()).isEqualTo("completed");
        verify(artifactMapper).listRefsByMessageIds(List.of(9L));
    }

    @Test
    @DisplayName("消息详情：assistant 补兄弟序号（1 起）与兄弟总数，一次按父查询避免 N+1")
    void getMessageFillsSiblingPositionForAssistant() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage msg = message(9L, "assistant", "second");
        msg.setParentMessageId(8L);
        when(messageMapper.selectById(9L)).thenReturn(msg);
        when(artifactMapper.listRefsByMessageIds(anyList())).thenReturn(List.of());
        when(messageMapper.listAssistantSiblingIds(1L, List.of(8L)))
                .thenReturn(List.of(siblingRow(8L, 7L), siblingRow(8L, 9L)));

        AiMessageVO vo = service.getMessage(9L, USER_ID, false);

        assertThat(vo.getSiblingIndex()).isEqualTo(2);
        assertThat(vo.getSiblingCount()).isEqualTo(2);
        verify(messageMapper).listAssistantSiblingIds(1L, List.of(8L));
    }

    @Test
    @DisplayName("消息详情：root 级 assistant 与非 assistant 不参与兄弟分组，字段为 null")
    void getMessageLeavesSiblingNullForRootAssistantAndUser() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage rootAssistant = message(9L, "assistant", "root");
        when(messageMapper.selectById(9L)).thenReturn(rootAssistant);
        when(artifactMapper.listRefsByMessageIds(anyList())).thenReturn(List.of());

        AiMessageVO assistantVo = service.getMessage(9L, USER_ID, false);

        SysAiMessage user = message(8L, "user", "ask");
        user.setParentMessageId(7L);
        when(messageMapper.selectById(8L)).thenReturn(user);
        AiMessageVO userVo = service.getMessage(8L, USER_ID, false);

        assertThat(assistantVo.getSiblingIndex()).isNull();
        assertThat(assistantVo.getSiblingCount()).isNull();
        assertThat(userVo.getSiblingIndex()).isNull();
        assertThat(userVo.getSiblingCount()).isNull();
        verify(messageMapper, never()).listAssistantSiblingIds(any(), anyList());
    }

    @Test
    @DisplayName("分支列表：只返回 assistant 子消息，user 子消息是下一轮提问不算分支")
    void getBranchesReturnsOnlyAssistantChildren() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectById(4L)).thenReturn(message(4L, "user", "ask"));
        when(messageMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(message(5L, "assistant", "a1"), message(6L, "assistant", "a2")));
        when(artifactMapper.listRefsByMessageIds(anyList())).thenReturn(List.of());

        List<AiMessageVO> branches = service.getBranches(1L, USER_ID, 4L);

        assertThat(branches).extracting(AiMessageVO::getId).containsExactly(5L, 6L);
        ArgumentCaptor<Wrapper<SysAiMessage>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(messageMapper).selectList(captor.capture());
        assertThat(captor.getValue().getSqlSegment()).contains("role");
        assertThat(((AbstractWrapper<?, ?, ?>) captor.getValue())
                        .getParamNameValuePairs()
                        .values())
                .contains("assistant");
    }

    @Test
    @DisplayName("切换分支：当前指针存在待恢复中断时拒绝切换（A0500），不移动指针")
    void switchBranchRejectedWhenCurrentPointerHasPendingInterrupt() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));
        when(stringRedisTemplate.hasKey("ai:interrupt:1:9")).thenReturn(true);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.switchBranch(1L, USER_ID, 5L));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.BUSINESS_ERROR);
        assertThat(ex.getMessage()).contains("中断确认");
        verify(conversationMapper, never()).updateCurrentBranch(anyLong(), any());
    }

    @Test
    @DisplayName("切换分支：无待恢复中断时正常切换")
    void switchBranchAllowedWithoutPendingInterrupt() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));
        when(stringRedisTemplate.hasKey("ai:interrupt:1:9")).thenReturn(false);

        AiConversationVO vo = service.switchBranch(1L, USER_ID, 5L);

        assertThat(vo.getCurrentBranchMessageId()).isEqualTo(5L);
        verify(conversationMapper).updateCurrentBranch(1L, 5L);
    }

    @Test
    @DisplayName("切换分支：指针为空时不查中断键（空会话/旧数据无挂起概念）")
    void switchBranchSkipsInterruptCheckWithoutPointer() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));

        service.switchBranch(1L, USER_ID, 5L);

        verify(stringRedisTemplate, never()).hasKey(anyString());
        verify(conversationMapper).updateCurrentBranch(1L, 5L);
    }

    @Test
    @DisplayName("切换分支：Redis 无句柄但库内 pending_interrupt 可续（awaiting_input 未过期）时拒绝")
    void switchBranchRejectsDbActionableInterruptWithoutRedisKey() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));
        when(messageMapper.selectById(9L))
                .thenReturn(awaitingMessage(9L, interrupt(LocalDateTime.now().plusMinutes(5))));
        when(stringRedisTemplate.hasKey("ai:interrupt:1:9")).thenReturn(false);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.switchBranch(1L, USER_ID, 5L));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.BUSINESS_ERROR);
        assertThat(ex.getMessage()).contains("中断确认");
        verify(conversationMapper, never()).updateCurrentBranch(anyLong(), any());
    }

    @Test
    @DisplayName("切换分支：库内中断已过期时放行（历史残留不挡切换）")
    void switchBranchAllowedWhenDbInterruptExpired() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));
        when(messageMapper.selectById(9L))
                .thenReturn(awaitingMessage(9L, interrupt(LocalDateTime.now().minusMinutes(1))));
        when(stringRedisTemplate.hasKey("ai:interrupt:1:9")).thenReturn(false);

        AiConversationVO vo = service.switchBranch(1L, USER_ID, 5L);

        assertThat(vo.getCurrentBranchMessageId()).isEqualTo(5L);
        verify(conversationMapper).updateCurrentBranch(1L, 5L);
    }

    @Test
    @DisplayName("切换分支：库内中断所在消息已离开 awaiting_input 时放行")
    void switchBranchAllowedWhenMessageLeftAwaitingInput() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        stubOwned(conv);
        when(messageMapper.selectById(5L)).thenReturn(message(5L, "assistant", "branch"));
        SysAiMessage resumed = message(9L, "assistant", "已恢复");
        resumed.setStatus("completed");
        resumed.setPendingInterrupt(interrupt(LocalDateTime.now().plusMinutes(5)));
        when(messageMapper.selectById(9L)).thenReturn(resumed);
        when(stringRedisTemplate.hasKey("ai:interrupt:1:9")).thenReturn(false);

        AiConversationVO vo = service.switchBranch(1L, USER_ID, 5L);

        assertThat(vo.getCurrentBranchMessageId()).isEqualTo(5L);
        verify(conversationMapper).updateCurrentBranch(1L, 5L);
    }

    @Test
    @DisplayName("待恢复中断：无中断返回 null")
    void pendingInterruptNullWhenAbsent() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectById(9L)).thenReturn(message(9L, "assistant", "x"));

        assertThat(service.getPendingInterrupt(9L, USER_ID, false)).isNull();
    }

    @Test
    @DisplayName("待恢复中断：消息 awaiting_input 且未过期时原样返回 actionable=true")
    void pendingInterruptKeepsActionableBeforeExpiry() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage msg = awaitingMessage(9L, interrupt(LocalDateTime.now().plusMinutes(5)));
        when(messageMapper.selectById(9L)).thenReturn(msg);

        AiContract.Interrupt interrupt = service.getPendingInterrupt(9L, USER_ID, false);

        assertThat(interrupt.actionable()).isTrue();
        assertThat(interrupt.type()).isEqualTo("confirm");
        assertThat(interrupt.confirmKind()).isEqualTo("tool_permission");
    }

    @Test
    @DisplayName("待恢复中断：超过 expiresAt 返回 actionable=false（不改写库内记录）")
    void pendingInterruptMarksExpiredNotActionable() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage msg = awaitingMessage(9L, interrupt(LocalDateTime.now().minusMinutes(1)));
        when(messageMapper.selectById(9L)).thenReturn(msg);

        AiContract.Interrupt interrupt = service.getPendingInterrupt(9L, USER_ID, false);

        assertThat(interrupt.actionable()).isFalse();
        assertThat(interrupt.id()).isEqualTo("int-1");
        assertThat(msg.getPendingInterrupt().actionable()).isTrue();
        verify(messageMapper, never()).updateById(any());
    }

    @Test
    @DisplayName("待恢复中断：带时区偏移的 expiresAt 同样参与过期判定")
    void pendingInterruptHandlesOffsetExpiresAt() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage msg = awaitingMessage(9L, interruptWithExpiresAt("2020-01-01T00:00:00+08:00"));
        when(messageMapper.selectById(9L)).thenReturn(msg);

        assertThat(service.getPendingInterrupt(9L, USER_ID, false).actionable()).isFalse();
    }

    @Test
    @DisplayName("待恢复中断：库内 yyyy-MM-dd HH:mm:ss 口径同样参与过期判定")
    void pendingInterruptHandlesSqlDateTimeFormat() {
        stubOwned(conv(1L, USER_ID, 0));
        SysAiMessage expired = awaitingMessage(9L, interruptWithExpiresAt("2020-01-01 00:00:00"));
        when(messageMapper.selectById(9L)).thenReturn(expired);
        assertThat(service.getPendingInterrupt(9L, USER_ID, false).actionable()).isFalse();

        SysAiMessage alive = awaitingMessage(8L, interruptWithExpiresAt("2099-01-01 00:00:00"));
        when(messageMapper.selectById(8L)).thenReturn(alive);
        assertThat(service.getPendingInterrupt(8L, USER_ID, false).actionable()).isTrue();
    }

    @Test
    @DisplayName("待恢复中断：消息已不是 awaiting_input 时 actionable=false（历史残留不给可续信号）")
    void pendingInterruptNotActionableWhenStatusLeftAwaitingInput() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(9L);
        SysAiMessage msg = message(9L, "assistant", "x");
        msg.setStatus("completed");
        msg.setPendingInterrupt(interrupt(LocalDateTime.now().plusMinutes(5)));
        stubActiveChain(conv, List.of(msg));

        AiContract.Interrupt interrupt = service.getPendingInterrupt(9L, USER_ID, false);
        AiContract.Interrupt inList = service.listMessages(1L, USER_ID, null, 50, false)
                .getList()
                .get(0)
                .getPendingInterrupt();

        assertThat(interrupt.actionable()).isFalse();
        assertThat(inList.actionable()).isFalse();
        // payload 保留：expiresAt/id 等仍原样返回，前端据此展示"不可续"
        assertThat(interrupt.id()).isEqualTo("int-1");
        assertThat(interrupt.expiresAt()).isEqualTo(msg.getPendingInterrupt().expiresAt());
    }

    @Test
    @DisplayName("过期中断：列表、详情、专用端点三处同口径（actionable=false，且不回写库）")
    void expiredInterruptNormalizedConsistentlyAcrossReadModels() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        SysAiMessage msg = awaitingMessage(5L, interrupt(LocalDateTime.now().minusDays(30)));
        stubActiveChain(conv, List.of(msg));

        AiContract.Interrupt inList = service.listMessages(1L, USER_ID, null, 50, false)
                .getList()
                .get(0)
                .getPendingInterrupt();
        AiContract.Interrupt inDetail = service.getMessage(5L, USER_ID, false).getPendingInterrupt();
        AiContract.Interrupt viaEndpoint = service.getPendingInterrupt(5L, USER_ID, false);

        assertThat(inList.actionable()).isFalse();
        assertThat(inDetail.actionable()).isFalse();
        assertThat(viaEndpoint.actionable()).isFalse();
        assertThat(inList.id()).isEqualTo("int-1");
        assertThat(inDetail.expiresAt()).isEqualTo(msg.getPendingInterrupt().expiresAt());
        // 读时归一化，库内记录保持中断发生时刻的事实
        assertThat(msg.getPendingInterrupt().actionable()).isTrue();
        verify(messageMapper, never()).updateById(any());
    }

    @Test
    @DisplayName("待恢复中断：消息不存在抛 A0401")
    void pendingInterruptRejectsMissingMessage() {
        stubOwned(conv(1L, USER_ID, 0));
        when(messageMapper.selectById(9L)).thenReturn(null);

        BusinessException ex =
                assertThrows(BusinessException.class, () -> service.getPendingInterrupt(9L, USER_ID, false));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
    }

    @Test
    @DisplayName("消息列表（admin 视角）：不做归属过滤、全量分页，total 为会话消息总数，兄弟字段为 null")
    void listMessagesAdminBypassesOwnership() {
        SysAiConversation foreign = conv(1L, 999L, 0);
        when(conversationMapper.selectById(1L)).thenReturn(foreign);
        SysAiMessage msg = message(5L, "assistant", "x");
        msg.setParentMessageId(4L);
        when(messageMapper.selectPage(any(Page.class), any(Wrapper.class))).thenReturn(messagePage(List.of(msg)));
        when(messageMapper.selectCount(any(Wrapper.class))).thenReturn(1L);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, null, 50, true);

        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L);
        assertThat(result.getTotal()).isEqualTo(1L);
        assertThat(result.getList().get(0).getSiblingIndex()).isNull();
        assertThat(result.getList().get(0).getSiblingCount()).isNull();
        verify(messageMapper, never()).listAssistantSiblingIds(any(), anyList());
    }

    @Test
    @DisplayName("消息列表：翻页按链上 id 游标，新消息只出现在最新页（不重不漏）")
    void listMessagesCursorStableUnderConcurrentInsert() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        List<SysAiMessage> store = new ArrayList<>();
        for (long id = 1; id <= 5; id++) {
            SysAiMessage msg = message(id, "user", "m" + id);
            msg.setParentMessageId(id == 1 ? null : id - 1);
            store.add(msg);
        }
        conv.setCurrentBranchMessageId(5L);
        stubOwned(conv);
        when(messageMapper.selectList(any(Wrapper.class))).thenAnswer(inv -> new ArrayList<>(store));
        when(messageMapper.selectById(anyLong()))
                .thenAnswer(inv -> store.stream()
                        .filter(m -> m.getId().equals(inv.getArgument(0, Long.class)))
                        .findFirst()
                        .orElse(null));

        AiMessagePageVO p1 = service.listMessages(1L, USER_ID, null, 2, false);
        assertThat(p1.getList()).extracting(AiMessageVO::getId).containsExactly(5L, 4L);
        assertThat(p1.isHasMore()).isTrue();
        assertThat(p1.getTotal()).isEqualTo(5L);

        // 翻页途中并发插入更新的消息（id=6）并成为新的激活链尾，不应污染后续游标页
        SysAiMessage added = message(6L, "assistant", "m6");
        added.setParentMessageId(5L);
        store.add(added);
        conv.setCurrentBranchMessageId(6L);

        Long cursor = p1.getList().get(p1.getList().size() - 1).getId();
        AiMessagePageVO p2 = service.listMessages(1L, USER_ID, cursor, 2, false);
        assertThat(p2.getList()).extracting(AiMessageVO::getId).containsExactly(3L, 2L);
        assertThat(p2.isHasMore()).isTrue();

        AiMessagePageVO p3 =
                service.listMessages(1L, USER_ID, p2.getList().get(1).getId(), 2, false);
        assertThat(p3.getList()).extracting(AiMessageVO::getId).containsExactly(1L);
        assertThat(p3.isHasMore()).isFalse();
        // total 为"链长"，随并发插入增长
        assertThat(p3.getTotal()).isEqualTo(6L);

        List<Long> merged = new ArrayList<>();
        merged.addAll(p1.getList().stream().map(AiMessageVO::getId).toList());
        merged.addAll(p2.getList().stream().map(AiMessageVO::getId).toList());
        merged.addAll(p3.getList().stream().map(AiMessageVO::getId).toList());
        assertThat(merged).containsExactly(5L, 4L, 3L, 2L, 1L).doesNotContain(6L);
    }

    @Test
    @DisplayName("消息列表：before 超出 Integer.MAX_VALUE 时按完整 long 参与链上过滤（不被截断）")
    void listMessagesBeforeAboveIntegerMaxNotTruncated() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(5L);
        List<SysAiMessage> chain = new ArrayList<>();
        for (long id = 1; id <= 5; id++) {
            SysAiMessage msg = message(id, "user", "m" + id);
            msg.setParentMessageId(id == 1 ? null : id - 1);
            chain.add(msg);
        }
        stubActiveChain(conv, chain);

        AiMessagePageVO result = service.listMessages(1L, USER_ID, (long) Integer.MAX_VALUE + 10L, 1, false);

        // 若 before 被截断成 int（负数），链上全部消息会被过滤掉
        assertThat(result.getList()).extracting(AiMessageVO::getId).containsExactly(5L);
        assertThat(result.getTotal()).isEqualTo(5L);
        assertThat(result.isHasMore()).isTrue();
    }

    @Test
    @DisplayName("会话搜索：关键词中的 \\ % _ 先转义再参与 LIKE（标题与消息正文同口径）")
    void listEscapesLikeWildcardsBeforeQuery() {
        stubOwned(conv(1L, USER_ID, 0));
        Page<SysAiConversation> page = new Page<>(1, 10, 1);
        page.setRecords(new ArrayList<>(List.of(conv(1L, USER_ID, 0))));
        when(conversationMapper.selectPage(any(Page.class), any(Wrapper.class))).thenReturn(page);

        AiConversationPageQuery query = new AiConversationPageQuery();
        query.setKeyword("50%_\\");
        service.list(USER_ID, query, false);

        // 消息正文检索：\ % _ 各转义一层
        verify(messageMapper).listConversationIdsByKeyword("50\\%\\_\\\\");
        verify(messageMapper).findLatestIdsByKeyword(List.of(1L), "50\\%\\_\\\\");

        // 标题检索：显式 ESCAPE，参数为 "%<escaped>%"
        ArgumentCaptor<Wrapper<SysAiConversation>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(conversationMapper).selectPage(any(Page.class), captor.capture());
        assertThat(captor.getValue().getSqlSegment()).contains("ESCAPE");
        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) captor.getValue();
        assertThat(wrapper.getParamNameValuePairs().values()).contains("%50\\%\\_\\\\%");
    }

    @Test
    @DisplayName("消息检索 SQL 显式声明 ESCAPE，服务层的转义符才会被 LIKE 解释")
    void messageSearchSqlDeclaresEscapeClause() {
        for (String methodName : List.of("findLatestIdsByKeyword", "listConversationIdsByKeyword")) {
            String sql = Arrays.stream(SysAiMessageMapper.class.getMethods())
                    .filter(method -> method.getName().equals(methodName))
                    .findFirst()
                    .map(method ->
                            String.join(" ", method.getAnnotation(Select.class).value()))
                    .orElseThrow(() -> new AssertionError("缺少方法 " + methodName));
            assertThat(sql).contains("ESCAPE '\\\\'");
        }
    }

    private static Page<SysAiMessage> messagePage(List<SysAiMessage> records) {
        Page<SysAiMessage> page = new Page<>(1, records.size());
        page.setRecords(new ArrayList<>(records));
        return page;
    }

    /**
     * 激活链桩：指针指向链尾时按 id 命中该消息；指针为空时回退到链上最大 id；
     * 链内容由 selectList 提供（模拟"本会话未删消息"）。
     */
    private void stubActiveChain(SysAiConversation conv, List<SysAiMessage> all) {
        stubOwned(conv);
        Long tailId = conv.getCurrentBranchMessageId();
        if (tailId != null) {
            all.stream()
                    .filter(m -> tailId.equals(m.getId()))
                    .findFirst()
                    .ifPresent(tail -> when(messageMapper.selectById(tailId)).thenReturn(tail));
        } else if (!all.isEmpty()) {
            Long maxId =
                    all.stream().map(SysAiMessage::getId).max(Long::compareTo).orElseThrow();
            when(messageMapper.getMaxMessageId(conv.getId())).thenReturn(maxId);
        }
        when(messageMapper.selectList(any(Wrapper.class))).thenReturn(all);
    }

    private static SysAiMessageMapper.SiblingRow siblingRow(Long parentId, Long messageId) {
        SysAiMessageMapper.SiblingRow row = new SysAiMessageMapper.SiblingRow();
        row.setParentMessageId(parentId);
        row.setMessageId(messageId);
        return row;
    }

    @Test
    @DisplayName("导出 markdown：正文由 text 块派生，思考/工具块与非对话角色不进导出")
    void exportMarkdownDerivesTextFromBlocks() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(3L);
        stubOwned(conv);
        SysAiMessage tool = message(2L, "tool", "工具输出不该导出");
        tool.setParentMessageId(1L);
        SysAiMessage assistant = message(3L, "assistant", "已处理");
        assistant.setBlocks(List.of(
                new AiContract.Block("b1", "thinking", 0L, "内部推理不该导出", 120L, null),
                new AiContract.Block("b2", "text", 0L, "已处理", null, null),
                new AiContract.Block(
                        "b3",
                        "tool_use",
                        0L,
                        null,
                        null,
                        new AiContract.ToolCall(
                                "call-1",
                                "predict",
                                Map.of("imageId", 1),
                                null,
                                null,
                                "success",
                                null,
                                30L,
                                null,
                                false))));
        assistant.setParentMessageId(2L);
        when(messageMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(message(1L, "user", "帮我处理图片"), tool, assistant));

        AiConversationService.ConversationExport export = service.export(1L, USER_ID, "markdown");

        assertThat(export.contentType()).isEqualTo("text/markdown");
        assertThat(export.content())
                .contains("帮我处理图片")
                .contains("已处理")
                .doesNotContain("内部推理不该导出")
                .doesNotContain("工具输出不该导出")
                .doesNotContain("predict");
    }

    @Test
    @DisplayName("导出 json：消息体仅含 role/content/create_time")
    void exportJsonContainsConversationAndMessages() {
        SysAiConversation conv = conv(1L, USER_ID, 0);
        conv.setCurrentBranchMessageId(2L);
        stubOwned(conv);
        when(messageMapper.selectList(any(Wrapper.class)))
                .thenReturn(List.of(message(1L, "user", "问题"), message(2L, "assistant", "回答")));

        AiConversationService.ConversationExport export = service.export(1L, USER_ID, "json");

        assertThat(export.contentType()).isEqualTo("application/json");
        assertThat(export.content())
                .contains("\"conversation\"")
                .contains("\"agent_code\"")
                .contains("\"messages\"");
    }

    @Test
    @DisplayName("回收站恢复：超出 30 天窗口报 A0401")
    void restoreOutsideWindowRejected() {
        when(conversationMapper.selectInTrash(eq(1L), eq(USER_ID), any(LocalDateTime.class)))
                .thenReturn(null);

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

    private SysAiMessage message(Long id, String role, String text) {
        SysAiMessage message = new SysAiMessage();
        message.setId(id);
        message.setConversationId(1L);
        message.setRole(role);
        message.setStatus("completed");
        message.setBlocks(
                text == null ? List.of() : List.of(new AiContract.Block("b-" + id, "text", 0L, text, null, null)));
        message.setInputTokens(12);
        message.setOutputTokens(34);
        message.setCachedInputTokens(5);
        message.setCredits(6L);
        message.setEdited(1);
        message.setUsedMemoryIds(List.of(7L, 8L));
        message.setSuggestions(List.of("继续", "换一个"));
        message.setCreateTime(LocalDateTime.now());
        return message;
    }

    /** 处于 awaiting_input 的消息（中断可续的前提） */
    private SysAiMessage awaitingMessage(Long id, AiContract.Interrupt interrupt) {
        SysAiMessage message = message(id, "assistant", "等待确认");
        message.setStatus("awaiting_input");
        message.setPendingInterrupt(interrupt);
        return message;
    }

    private static SysAiArtifactMapper.ArtifactRefRow artifactRef(Long messageId, Long artifactId) {
        SysAiArtifactMapper.ArtifactRefRow row = new SysAiArtifactMapper.ArtifactRefRow();
        row.setMessageId(messageId);
        row.setId(artifactId);
        return row;
    }

    private static AiContract.Plan plan() {
        return new AiContract.Plan(
                "analyze",
                "running",
                List.of(new AiContract.PlanTask("t1", "拆解问题", List.of(), "running", "react")),
                List.of(new AiContract.PlanRevision(1L, "初始计划", "user", "2026-10-21T10:00:00")),
                false);
    }

    private static AiContract.Interrupt interrupt(LocalDateTime expiresAt) {
        return interruptWithExpiresAt(expiresAt.toString());
    }

    private static AiContract.Interrupt interruptWithExpiresAt(String expiresAt) {
        return new AiContract.Interrupt(
                "int-1",
                "confirm",
                "tool_permission",
                "predict",
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                "需要确认工具权限",
                "调用预测服务前需授权",
                "2026-10-21T10:00:00",
                expiresAt,
                true);
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
