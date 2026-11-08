package com.pei.clarity.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.common.util.AiDateTimeUtils;
import com.pei.clarity.mapper.AiInsightMapper;
import com.pei.clarity.mapper.SysAiAgentMapper;
import com.pei.clarity.mapper.SysAiAgentVersionMapper;
import com.pei.clarity.mapper.SysAiArtifactMapper;
import com.pei.clarity.mapper.SysAiConversationMapper;
import com.pei.clarity.mapper.SysAiMessageMapper;
import com.pei.clarity.model.dto.ai.AiContract;
import com.pei.clarity.model.entity.SysAiAgent;
import com.pei.clarity.model.entity.SysAiAgentVersion;
import com.pei.clarity.model.entity.SysAiConversation;
import com.pei.clarity.model.entity.SysAiMessage;
import com.pei.clarity.model.form.AiConversationBatchForm;
import com.pei.clarity.model.form.AiConversationCreateForm;
import com.pei.clarity.model.form.AiConversationUpdateForm;
import com.pei.clarity.model.query.AiConversationPageQuery;
import com.pei.clarity.model.read.AnomalyStatusRead;
import com.pei.clarity.model.read.ConversationConsumptionRead;
import com.pei.clarity.model.read.DisplayNameRead;
import com.pei.clarity.model.vo.AiConversationVO;
import com.pei.clarity.model.vo.AiMessagePageVO;
import com.pei.clarity.model.vo.AiMessageVO;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * AI 会话与消息服务（非推理类端点）。
 *
 * <p>行为对齐 clarity-python {@code ai_conversation_service}：软删方案 A（deleted=id + delete_time）、
 * 回收站 30 天窗口、置顶上限与用户级锁、消息分页取最新一页。
 *
 * @author clarity
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiConversationService {

    /** 未指定 Agent 时的平台默认 Agent 编码 */
    private static final String DEFAULT_AGENT_CODE = "default";

    /** 消息可续状态：只有该状态下残留的中断才可能 actionable */
    private static final String STATUS_AWAITING_INPUT = "awaiting_input";

    /** 会话标题 LIKE（显式 ESCAPE，与消息检索、Python/Go 的转义口径一致） */
    private static final String TITLE_LIKE = "title LIKE {0} ESCAPE '\\\\'";

    /** 推理中断句柄键前缀（与 python {@code InterruptHandler.INTERRUPT_KEY} 同口径：{convId}:{msgId}） */
    private static final String INTERRUPT_KEY_PREFIX = "ai:interrupt:";

    /** 置顶会话上限 */
    private static final int PINNED_CONVERSATION_LIMIT = 10;

    /** 软删除恢复窗口（天） */
    private static final int RECYCLE_WINDOW_DAYS = 30;

    private static final Duration PIN_LOCK_TTL = Duration.ofSeconds(10);

    /** 置顶锁释放：仅当值仍是本次 token 时删除，避免误删他人锁 */
    private static final DefaultRedisScript<Long> RELEASE_LOCK_SCRIPT = new DefaultRedisScript<>(
            "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
            Long.class);

    private static final Map<String, String> SCENE_PROMPTS = Map.of(
            "general",
            "【角色】你是用户亲切、可靠的对话助手。\n【任务】回应用户的一般性提问与闲聊，给出清晰、准确的解答。\n"
                    + "【指令】先理解意图再作答；不确定时坦承并给出进一步澄清；涉及专业领域时提供必要的背景。\n"
                    + "【格式】分点或分段组织回答，语言简洁友好。",
            "image_dispatch",
            "【角色】你是图像处理任务的调度专家。\n【任务】接收用户的图像处理请求，识别处理目标并调度合适的算法完成处理。\n"
                    + "【指令】明确输入与期望输出；选择合适的处理算法与参数；处理结果仅以产物引用形式反馈，不展开处理过程细节。\n"
                    + "【格式】先复述任务理解，再说明所选算法与参数，最后给出结果引用。",
            "multi_step",
            "【角色】你是擅长拆解复杂任务的推理专家。\n【任务】将复杂问题分解为若干可执行的步骤，逐步求解并给出最终结论。\n"
                    + "【指令】先规划步骤再执行；每步说明依据；遇到依赖前置结果的步骤须先取得结果；避免跳跃式推断。\n"
                    + "【格式】以编号步骤呈现推理过程，最后以「结论」区块汇总。",
            "algorithm_recommend",
            "【角色】你是图像处理算法的推荐顾问。\n【任务】根据用户提供的图像特征与处理诉求，推荐最合适的算法及参数。\n"
                    + "【指令】结合用户偏好与历史处理习惯给出推荐；说明推荐理由与适用场景；提供备选方案。\n"
                    + "【格式】列出推荐算法（含理由与匹配度），再给出参数建议与备选。",
            "scheduled_task",
            "【角色】你是可靠的任务编排与定时调度助手。\n【任务】帮助用户设定、调整、查询定时处理任务，并确认任务已正确配置。\n"
                    + "【指令】明确任务内容、执行频率与目标对象；校验参数合法性；反馈任务创建/变更结果。\n"
                    + "【格式】以任务概览形式列出任务要素（内容/频率/状态）。");

    private final SysAiConversationMapper conversationMapper;

    private final SysAiMessageMapper messageMapper;

    private final SysAiArtifactMapper artifactMapper;

    private final SysAiAgentMapper agentMapper;

    private final SysAiAgentVersionMapper agentVersionMapper;

    private final AiInsightMapper insightMapper;

    private final AiDefaultModelResolver defaultModelResolver;

    private final StringRedisTemplate stringRedisTemplate;

    private final ObjectMapper objectMapper;

    /**
     * 会话导出结果（控制器直接写响应体）
     */
    public record ConversationExport(String filename, String contentType, String content) {}

    // ── 会话 CRUD ─────────────────────────────────────────────

    @Transactional
    public AiConversationVO create(Long userId, AiConversationCreateForm form) {
        String scene =
                form.getScene() != null && SCENE_PROMPTS.containsKey(form.getScene()) ? form.getScene() : "general";
        AgentAnchor anchor = resolveAgentAnchor(form.getAgentCode());
        SysAiConversation conv = new SysAiConversation();
        conv.setUserId(userId);
        conv.setTitle(form.getTitle() != null && !form.getTitle().isBlank() ? form.getTitle() : "新对话");
        conv.setModel(
                form.getModel() != null ? form.getModel() : defaultModelResolver.resolveForUser(userId, anchor.code()));
        conv.setAgentCode(anchor.code());
        conv.setAgentVersion(anchor.version());
        conv.setSystemPrompt(form.getSystemPrompt() != null ? form.getSystemPrompt() : SCENE_PROMPTS.get(scene));
        conv.setModelConfig(form.getModelConfig());
        conv.setSuggestionsEnabled(Boolean.FALSE.equals(form.getSuggestionsEnabled()) ? 0 : 1);
        conv.setApiKeyId(form.getApiKeyId());
        conv.setStatus(1);
        conv.setPinned(0);
        conv.setMessageCount(0);
        conv.setTitleSource("auto");
        conversationMapper.insert(conv);
        return toVO(conv);
    }

    public IPage<AiConversationVO> list(Long userId, AiConversationPageQuery query, boolean admin) {
        Integer status = query.getStatus();
        Integer statusFilter;
        if (admin) {
            statusFilter = status == null || status == 0 ? null : status;
        } else {
            statusFilter = status == null ? 1 : (status == 0 ? null : status);
        }
        boolean keywordPresent =
                query.getKeyword() != null && !query.getKeyword().isBlank();
        LambdaQueryWrapper<SysAiConversation> wrapper = new LambdaQueryWrapper<SysAiConversation>()
                .eq(!admin, SysAiConversation::getUserId, userId)
                .eq(statusFilter != null, SysAiConversation::getStatus, statusFilter)
                .orderByDesc(SysAiConversation::getPinned)
                .orderByDesc(SysAiConversation::getPinnedAt)
                .orderByDesc(SysAiConversation::getLastMessageAt)
                .orderByDesc(SysAiConversation::getId);
        if (keywordPresent) {
            String keyword = query.getKeyword();
            // 关键词是用户输入，LIKE 模式须先转义 \ % _，否则搜 "50%" / "a_b" 会退化成通配符
            if (admin) {
                wrapper.and(w -> w.apply(TITLE_LIKE, likePattern(keyword)));
            } else {
                // 无 ES 全文检索能力，以标题 LIKE + 消息正文命中等价定位（命中消息 ID 供前端定位）
                List<Long> hitConvIds = messageMapper.listConversationIdsByKeyword(escapeLike(keyword));
                if (hitConvIds.isEmpty()) {
                    wrapper.and(w -> w.apply(TITLE_LIKE, likePattern(keyword)));
                } else {
                    wrapper.and(w ->
                            w.apply(TITLE_LIKE, likePattern(keyword)).or().in(SysAiConversation::getId, hitConvIds));
                }
            }
        }
        Page<SysAiConversation> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiConversation> convPage = conversationMapper.selectPage(page, wrapper);
        List<AiConversationVO> records = new ArrayList<>();
        for (SysAiConversation conv : convPage.getRecords()) {
            records.add(admin ? toVO(conv) : toVOWithUnread(conv));
        }
        if (admin) {
            attachAuditFields(records);
        } else if (keywordPresent) {
            attachMatchedMessages(records, query.getKeyword());
        }
        return pageOf(page, records, convPage.getTotal());
    }

    public AiConversationVO getDetail(Long convId, Long userId, boolean admin) {
        if (admin) {
            SysAiConversation conv = conversationMapper.selectById(convId);
            if (conv == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
            }
            AiConversationVO vo = toVOWithUnread(conv);
            attachAuditFields(new ArrayList<>(List.of(vo)));
            return vo;
        }
        return toVO(getOwned(convId, userId));
    }

    @Transactional
    public AiConversationVO update(Long convId, Long userId, AiConversationUpdateForm form) {
        SysAiConversation conv = getOwned(convId, userId);
        boolean wasPinned = Integer.valueOf(1).equals(conv.getPinned());
        if (form.getTitle() != null) {
            conv.setTitle(form.getTitle());
            conv.setTitleSource("manual");
        }
        if (form.getModel() != null) {
            conv.setModel(form.getModel());
        }
        if (form.getSystemPrompt() != null) {
            conv.setSystemPrompt(form.getSystemPrompt());
        }
        if (form.getModelConfig() != null) {
            conv.setModelConfig(form.getModelConfig());
        }
        if (form.getSuggestionsEnabled() != null) {
            conv.setSuggestionsEnabled(Boolean.TRUE.equals(form.getSuggestionsEnabled()) ? 1 : 0);
        }
        if (form.getStatus() != null) {
            conv.setStatus(form.getStatus());
        }
        if (form.getAgentCode() != null) {
            AgentAnchor anchor = resolveAgentAnchor(form.getAgentCode());
            conv.setAgentCode(anchor.code());
            conv.setAgentVersion(anchor.version());
            // 切换 Agent 且未同时显式指定模型时，会话模型按新 Agent 重新解析
            // （否则"换了 Agent 模型不变"，与新建会话的解析口径不一致）
            if (form.getModel() == null) {
                conv.setModel(defaultModelResolver.resolveForUser(userId, anchor.code()));
            }
        }
        if (form.getPinned() != null) {
            if (Integer.valueOf(1).equals(form.getPinned())) {
                if (!wasPinned) {
                    conv.setPinned(1);
                    conv.setPinnedAt(pinWithLimit(convId, userId));
                }
            } else {
                conv.setPinned(0);
                conv.setPinnedAt(null);
                conversationMapper.setPinned(convId, 0, null);
            }
        }
        conversationMapper.updateById(conv);
        return toVOWithUnread(conv);
    }

    @Transactional
    public void delete(Long convId, Long userId) {
        SysAiConversation conv = getOwned(convId, userId);
        conversationMapper.softDeleteByIds(List.of(conv.getId()));
    }

    @Transactional
    public AiConversationVO restore(Long convId, Long userId) {
        SysAiConversation conv = conversationMapper.selectInTrash(
                convId, userId, LocalDateTime.now().minusDays(RECYCLE_WINDOW_DAYS));
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已超出恢复窗口");
        }
        conversationMapper.restoreByIds(List.of(conv.getId()));
        conv.setDeleted(0L);
        conv.setDeleteTime(null);
        return toVOWithUnread(conv);
    }

    public IPage<AiConversationVO> listTrash(Long userId, int pageNum, int pageSize) {
        Page<SysAiConversation> page = new Page<>(pageNum, pageSize);
        IPage<SysAiConversation> trash = conversationMapper.selectTrashPage(
                page, userId, LocalDateTime.now().minusDays(RECYCLE_WINDOW_DAYS));
        List<AiConversationVO> records =
                trash.getRecords().stream().map(this::toVOWithUnread).toList();
        return pageOf(page, records, trash.getTotal());
    }

    @Transactional
    public int batchOperate(Long userId, AiConversationBatchForm form) {
        int count = 0;
        for (Long convId : form.getIds()) {
            SysAiConversation conv = getOwned(convId, userId);
            switch (form.getAction()) {
                case "archive" -> {
                    if (!Integer.valueOf(1).equals(conv.getStatus())) {
                        throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅活跃会话可归档");
                    }
                    conversationMapper.updateStatusByIds(List.of(conv.getId()), 2);
                }
                case "restore" -> {
                    if (!Integer.valueOf(2).equals(conv.getStatus())) {
                        throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅已归档会话可恢复");
                    }
                    conversationMapper.updateStatusByIds(List.of(conv.getId()), 1);
                }
                case "delete" -> {
                    if (!Boolean.TRUE.equals(form.getConfirm())) {
                        throw new BusinessException(ResultCode.PARAM_ERROR, "批量删除需二次确认");
                    }
                    conversationMapper.softDeleteByIds(List.of(conv.getId()));
                }
                default -> throw new BusinessException(ResultCode.PARAM_ERROR, "不支持的批量操作类型: " + form.getAction());
            }
            count++;
        }
        return count;
    }

    @Transactional
    public AiConversationVO pin(Long convId, Long userId) {
        SysAiConversation conv = getOwned(convId, userId);
        if (!Integer.valueOf(1).equals(conv.getPinned())) {
            conv.setPinned(1);
            conv.setPinnedAt(pinWithLimit(convId, userId));
        }
        return toVOWithUnread(conv);
    }

    @Transactional
    public AiConversationVO unpin(Long convId, Long userId) {
        SysAiConversation conv = getOwned(convId, userId);
        conversationMapper.setPinned(convId, 0, null);
        conv.setPinned(0);
        conv.setPinnedAt(null);
        return toVOWithUnread(conv);
    }

    @Transactional
    public AiConversationVO markRead(Long convId, Long userId) {
        SysAiConversation conv = getOwned(convId, userId);
        Long lastMsgId = messageMapper.getLastMessageId(convId);
        if (lastMsgId != null) {
            conversationMapper.markRead(convId, lastMsgId);
            conv.setLastReadMessageId(lastMsgId);
        }
        return toVOWithUnread(conv);
    }

    /**
     * 导出会话：沿当前激活分支回溯全部消息，仅导 user/assistant 正文（思考与工具块不导出）
     */
    public ConversationExport export(Long convId, Long userId, String format) {
        SysAiConversation conv = getOwned(convId, userId);
        List<SysAiMessage> chain = new ArrayList<>();
        if (conv.getCurrentBranchMessageId() != null) {
            chain = chainByTail(convId, conv.getCurrentBranchMessageId()).stream()
                    .filter(m -> "user".equals(m.getRole()) || "assistant".equals(m.getRole()))
                    .toList();
        }
        String fmt = format == null ? "markdown" : format;
        if ("json".equals(fmt)) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put(
                    "conversation",
                    Map.of(
                            "id", conv.getId(),
                            "title", conv.getTitle() == null ? "" : conv.getTitle(),
                            "model", conv.getModel() == null ? "" : conv.getModel(),
                            "agent_code", conv.getAgentCode() == null ? "" : conv.getAgentCode(),
                            "create_time",
                                    conv.getCreateTime() == null
                                            ? ""
                                            : conv.getCreateTime().toString()));
            List<Map<String, Object>> messages = new ArrayList<>();
            for (SysAiMessage msg : chain) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("role", msg.getRole());
                item.put("content", textOf(msg));
                item.put(
                        "create_time",
                        msg.getCreateTime() == null ? "" : msg.getCreateTime().toString());
                messages.add(item);
            }
            payload.put("messages", messages);
            try {
                return new ConversationExport(
                        "conversation_" + convId + ".json",
                        "application/json",
                        objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(payload));
            } catch (Exception e) {
                throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "会话导出失败");
            }
        }
        StringBuilder sb = new StringBuilder("# ").append(conv.getTitle()).append("\n\n");
        for (SysAiMessage msg : chain) {
            sb.append("## ")
                    .append("user".equals(msg.getRole()) ? "用户" : "助手")
                    .append("\n\n")
                    .append(textOf(msg))
                    .append("\n\n");
        }
        return new ConversationExport("conversation_" + convId + ".md", "text/markdown", sb.toString());
    }

    // ── 会话消息 ─────────────────────────────────────────────

    /**
     * 会话消息列表。
     *
     * <p>self 视角（默认）只返回当前激活链：以会话 {@code currentBranchMessageId} 为链尾沿
     * {@code parentMessageId} 回溯（指针为空/悬空/指向已删消息时回退到会话最大 id 的未删消息），
     * 链即用户实际看到的对话。分页在链上切：{@code before} 之前的最后 {@code limit} 条按 id 倒序
     * 返回，{@code hasMore} 表示链上是否还有更早消息，{@code total} 为链长——分支消息不再混入列表。
     *
     * <p>admin 视角保持全量分支游标分页（审计需要看到所有分叉），{@code total} 为会话消息总数；
     * {@code siblingIndex/siblingCount} 仅在 self 视角的 assistant 消息上填充。
     */
    public AiMessagePageVO listMessages(Long convId, Long userId, Long before, int limit, boolean admin) {
        SysAiConversation conv = admin ? conversationMapper.selectById(convId) : getOwned(convId, userId);
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
        }
        if (admin) {
            return listAllBranchMessages(convId, before, limit);
        }
        List<SysAiMessage> chain = loadActiveChain(conv);
        List<SysAiMessage> eligible = before == null
                ? chain
                : chain.stream().filter(m -> m.getId() < before).toList();
        boolean hasMore = eligible.size() > limit;
        // 链回溯按 id 升序，接口契约仍按 id 倒序返回（最新一页在前）
        List<SysAiMessage> page =
                new ArrayList<>(eligible.subList(Math.max(0, eligible.size() - limit), eligible.size()));
        Collections.reverse(page);
        Map<Long, SiblingPosition> positions = siblingPositions(convId, page);
        Map<Long, List<Long>> artifactIds =
                artifactIdsByMessage(page.stream().map(SysAiMessage::getId).toList());
        List<AiMessageVO> records = new ArrayList<>();
        for (SysAiMessage msg : page) {
            SiblingPosition position = positions.get(msg.getId());
            records.add(toMessageVO(
                    msg,
                    artifactIds.getOrDefault(msg.getId(), List.of()),
                    position == null ? null : position.index(),
                    position == null ? null : position.count()));
        }
        AiMessagePageVO result = new AiMessagePageVO();
        result.setList(records);
        result.setTotal(chain.size());
        result.setHasMore(hasMore);
        return result;
    }

    /**
     * 管理端审计视角：全量分支游标分页（id 倒序，多取一条判定 hasMore），total 为会话消息总数
     */
    private AiMessagePageVO listAllBranchMessages(Long convId, Long before, int limit) {
        Page<SysAiMessage> cursor = new Page<>(1, limit + 1L, false);
        List<SysAiMessage> rows = messageMapper
                .selectPage(
                        cursor,
                        new LambdaQueryWrapper<SysAiMessage>()
                                .eq(SysAiMessage::getConversationId, convId)
                                .lt(before != null, SysAiMessage::getId, before)
                                .orderByDesc(SysAiMessage::getId))
                .getRecords();
        boolean hasMore = rows.size() > limit;
        List<SysAiMessage> pageRecords = hasMore ? rows.subList(0, limit) : rows;
        long total = messageMapper.selectCount(
                new LambdaQueryWrapper<SysAiMessage>().eq(SysAiMessage::getConversationId, convId));
        Map<Long, List<Long>> artifactIds = artifactIdsByMessage(
                pageRecords.stream().map(SysAiMessage::getId).toList());
        List<AiMessageVO> records = new ArrayList<>();
        for (SysAiMessage msg : pageRecords) {
            records.add(toMessageVO(msg, artifactIds.getOrDefault(msg.getId(), List.of())));
        }
        AiMessagePageVO result = new AiMessagePageVO();
        result.setList(records);
        result.setTotal(total);
        result.setHasMore(hasMore);
        return result;
    }

    /**
     * 当前激活链（id 升序）：指针为空/悬空/指向已删消息时回退到最新未删消息。
     *
     * <p>软删中间节点会在链上断开（{@code chainByTail} 只读取未删消息），此时只返回断点之后的活跃后缀。
     */
    private List<SysAiMessage> loadActiveChain(SysAiConversation conv) {
        Long tailId = conv.getCurrentBranchMessageId();
        if (tailId != null) {
            SysAiMessage tail = messageMapper.selectById(tailId);
            // 悬空、已软删（selectById 过滤 deleted=0）或不属于本会话，一律回退
            if (tail == null || !Objects.equals(tail.getConversationId(), conv.getId())) {
                tailId = null;
            }
        }
        if (tailId == null) {
            tailId = messageMapper.getMaxMessageId(conv.getId());
        }
        if (tailId == null) {
            return List.of();
        }
        return chainByTail(conv.getId(), tailId);
    }

    /**
     * 批量计算本页 assistant 消息的兄弟序号：{messageId: (1 起序号, 兄弟总数)}。
     *
     * <p>仅 parentMessageId 非空（分叉下的助手回复）参与，root 级与非 assistant 保持 null；
     * 一次 IN 查询取全部兄弟，避免逐条消息查询的 N+1。
     */
    private Map<Long, SiblingPosition> siblingPositions(Long convId, List<SysAiMessage> msgs) {
        Set<Long> pageIds = msgs.stream().map(SysAiMessage::getId).collect(Collectors.toSet());
        List<Long> parentIds = msgs.stream()
                .filter(m -> "assistant".equals(m.getRole()) && m.getParentMessageId() != null)
                .map(SysAiMessage::getParentMessageId)
                .distinct()
                .sorted()
                .toList();
        if (parentIds.isEmpty()) {
            return Map.of();
        }
        Map<Long, List<Long>> grouped = new LinkedHashMap<>();
        for (SysAiMessageMapper.SiblingRow row : messageMapper.listAssistantSiblingIds(convId, parentIds)) {
            grouped.computeIfAbsent(row.getParentMessageId(), k -> new ArrayList<>())
                    .add(row.getMessageId());
        }
        Map<Long, SiblingPosition> positions = new HashMap<>();
        for (List<Long> siblingIds : grouped.values()) {
            int count = siblingIds.size();
            for (int i = 0; i < count; i++) {
                if (pageIds.contains(siblingIds.get(i))) {
                    positions.put(siblingIds.get(i), new SiblingPosition(i + 1, count));
                }
            }
        }
        return positions;
    }

    /**
     * 消息详情。assistant 消息与列表同口径补 {@code siblingIndex/siblingCount}（同父未删 assistant，
     * id 升序，1 起）；root 级与非 assistant 保持 null。
     */
    public AiMessageVO getMessage(Long msgId, Long userId, boolean admin) {
        SysAiMessage msg = admin ? messageMapper.selectById(msgId) : getOwnedMessage(msgId, userId);
        if (msg == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        Map<Long, SiblingPosition> positions = siblingPositions(msg.getConversationId(), List.of(msg));
        SiblingPosition position = positions.get(msgId);
        return toMessageVO(
                msg,
                artifactIdsByMessage(List.of(msgId)).getOrDefault(msgId, List.of()),
                position == null ? null : position.index(),
                position == null ? null : position.count());
    }

    /**
     * 消息待恢复中断：无中断返回 null；消息已不在 {@code awaiting_input} 或超过 {@code expiresAt}
     * 时以 {@code actionable=false} 明确标注不可续（payload 原样保留，供前端展示"不可续"）。
     *
     * <p>过期判定在读取时按当前时间计算，不改写库内记录；列表、详情与本端点共用同一口径。
     */
    public AiContract.Interrupt getPendingInterrupt(Long msgId, Long userId, boolean admin) {
        SysAiMessage msg = admin ? messageMapper.selectById(msgId) : getOwnedMessage(msgId, userId);
        if (msg == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        return normalizePendingInterrupt(msg);
    }

    public List<AiMessageVO> getBranches(Long convId, Long userId, Long msgId) {
        getOwned(convId, userId);
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null || !Objects.equals(msg.getConversationId(), convId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        // 分支即父消息下的多个助手回复：user 子消息是下一轮提问，不属于可切换分支
        List<SysAiMessage> branches = messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, convId)
                .eq(SysAiMessage::getParentMessageId, msgId)
                .eq(SysAiMessage::getRole, "assistant")
                .orderByDesc(SysAiMessage::getCreateTime)
                .orderByDesc(SysAiMessage::getId));
        Map<Long, List<Long>> artifactIds =
                artifactIdsByMessage(branches.stream().map(SysAiMessage::getId).toList());
        return branches.stream()
                .map(branch -> toMessageVO(branch, artifactIds.getOrDefault(branch.getId(), List.of())))
                .toList();
    }

    /**
     * 切换当前激活分支。
     *
     * <p>当前指针存在待恢复中断时拒绝切换：切换会改变会话上下文，与 send/regenerate 的中断挂起校验同口径，
     * 避免闯入挂起流。中断判定同时看推理侧 Redis 句柄与库内 {@code pending_interrupt}（事实源），
     * 后者按 {@code actionable} 口径归一化后判定。
     */
    @Transactional
    public AiConversationVO switchBranch(Long convId, Long userId, Long msgId) {
        SysAiConversation conv = getOwned(convId, userId);
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null || !Objects.equals(msg.getConversationId(), convId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        if (hasPendingInterrupt(conv.getId(), conv.getCurrentBranchMessageId())) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止");
        }
        conversationMapper.updateCurrentBranch(convId, msgId);
        conv.setCurrentBranchMessageId(msgId);
        return toVO(conv);
    }

    /**
     * 删除助手消息（软删）。
     *
     * <p>删除策略：
     * <ul>
     *   <li>仍有未删子消息（非叶子）一律拒绝：中间节点软删会让后续消息失去父链、历史不可读，须先删后续对话；</li>
     *   <li>被删消息恰为当前分支指针时，指针回退到其父消息（可为 null），避免指针悬空；</li>
     *   <li>非激活分支上的旧消息允许删除，指针不动。</li>
     * </ul>
     *
     * <p>并发：children 检查、软删与指针回退要求与会话内的 send/switch 串行。事务内先
     * {@code SELECT ... FOR UPDATE} 锁会话行，再执行上述检查与写入；指针回退用 CAS，避免窗口内被
     * 并发 send 推进的指针被覆盖。
     */
    @Transactional
    public void deleteMessage(Long msgId, Long userId) {
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        if (!"assistant".equals(msg.getRole())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可删除");
        }
        SysAiConversation conv = conversationMapper.selectByIdForUpdate(msg.getConversationId(), userId);
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
        }
        Long childCount = messageMapper.selectCount(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, msg.getConversationId())
                .eq(SysAiMessage::getParentMessageId, msg.getId()));
        if (childCount != null && childCount > 0) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "请先删除该回复之后的对话");
        }
        messageMapper.softDeleteByIds(List.of(msg.getId()));
        if (Objects.equals(conv.getCurrentBranchMessageId(), msg.getId())) {
            conversationMapper.compareAndSetCurrentBranch(conv.getId(), msg.getId(), msg.getParentMessageId());
        }
    }

    // ── 内部工具 ─────────────────────────────────────────────

    /**
     * 会话锚定的 Agent 编码与已发布版本号（无已发布版本时为 null）
     */
    private record AgentAnchor(String code, Integer version) {}

    /**
     * assistant 消息的兄弟序号（1 起）与同父兄弟总数
     */
    private record SiblingPosition(int index, int count) {}

    private AgentAnchor resolveAgentAnchor(String agentCode) {
        String code = agentCode == null || agentCode.isBlank() ? DEFAULT_AGENT_CODE : agentCode.trim();
        SysAiAgent agent = agentMapper.selectOne(new LambdaQueryWrapper<SysAiAgent>()
                .eq(SysAiAgent::getAgentCode, code)
                .last("LIMIT 1"));
        if (agent == null) {
            return new AgentAnchor(code, null);
        }
        SysAiAgentVersion published = agentVersionMapper.getLatestPublished(agent.getId());
        return new AgentAnchor(code, published == null ? null : published.getVersionNo());
    }

    /** 会话归属校验：不存在或非本人（含已软删）抛 A0401 */
    public SysAiConversation getOwned(Long convId, Long userId) {
        SysAiConversation conv = conversationMapper.selectOne(new LambdaQueryWrapper<SysAiConversation>()
                .eq(SysAiConversation::getId, convId)
                .eq(SysAiConversation::getUserId, userId));
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
        }
        return conv;
    }

    /**
     * 过滤出归属于该用户的会话 ID 集合（产物等按会话归属反查时批量校验）
     */
    public Set<Long> listOwnedIds(Set<Long> convIds, Long userId) {
        if (convIds == null || convIds.isEmpty()) {
            return Set.of();
        }
        return conversationMapper
                .selectList(new LambdaQueryWrapper<SysAiConversation>()
                        .select(SysAiConversation::getId)
                        .eq(SysAiConversation::getUserId, userId)
                        .in(SysAiConversation::getId, convIds))
                .stream()
                .map(SysAiConversation::getId)
                .collect(Collectors.toSet());
    }

    /** 消息归属校验：消息表无 user_id，归属由所属会话判定（对齐 python get_by_id_and_user 的 join 语义） */
    public SysAiMessage getOwnedMessage(Long msgId, Long userId) {
        SysAiMessage msg = messageMapper.selectById(msgId);
        if (msg == null) {
            return null;
        }
        Long count = conversationMapper.selectCount(new LambdaQueryWrapper<SysAiConversation>()
                .eq(SysAiConversation::getId, msg.getConversationId())
                .eq(SysAiConversation::getUserId, userId));
        return count != null && count > 0 ? msg : null;
    }

    /**
     * 置顶会话并占用名额：用户级锁内完成"校验上限 + 置顶写入"，避免并发置顶双通过导致超限
     */
    private LocalDateTime pinWithLimit(Long convId, Long userId) {
        String lockKey = "ai:conv:pin:" + userId;
        String token = UUID.randomUUID().toString();
        Boolean locked = stringRedisTemplate.opsForValue().setIfAbsent(lockKey, token, PIN_LOCK_TTL);
        if (!Boolean.TRUE.equals(locked)) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "置顶操作并发冲突，请稍后再试");
        }
        try {
            if (conversationMapper.countActivePinned(userId) >= PINNED_CONVERSATION_LIMIT) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "置顶会话已达上限");
            }
            LocalDateTime now = LocalDateTime.now();
            conversationMapper.setPinned(convId, 1, now);
            return now;
        } finally {
            stringRedisTemplate.execute(RELEASE_LOCK_SCRIPT, List.of(lockKey), token);
        }
    }

    /**
     * 当前激活分支末端是否挂起待恢复中断。
     *
     * <p>两个信号取并集：推理侧（python {@code InterruptHandler}）写入的 Redis 句柄
     * {@code ai:interrupt:{convId}:{msgId}}（TTL 30 天），以及库内消息的 {@code pending_interrupt}
     * （事实源）。库内记录按 {@link #normalizePendingInterrupt} 归一化，仅 {@code actionable=true}
     * （awaiting_input 且未过期）才算挂起——避免历史残留/已过期中断挡住分支切换。
     *
     * <p>Java 侧只读不写中断状态：send/regenerate 全量转发推理服务，本地没有中断处理器。
     */
    private boolean hasPendingInterrupt(Long convId, Long tailMsgId) {
        if (tailMsgId == null) {
            return false;
        }
        if (Boolean.TRUE.equals(stringRedisTemplate.hasKey(INTERRUPT_KEY_PREFIX + convId + ":" + tailMsgId))) {
            return true;
        }
        SysAiMessage tail = messageMapper.selectById(tailMsgId);
        if (tail == null || !Objects.equals(tail.getConversationId(), convId)) {
            return false;
        }
        AiContract.Interrupt interrupt = normalizePendingInterrupt(tail);
        return interrupt != null && interrupt.actionable();
    }

    /**
     * 按当前激活分支末端回溯完整消息链（一次查询本会话消息 + 内存组链，visited 防环），返回正序链
     */
    private List<SysAiMessage> chainByTail(Long convId, Long tailMsgId) {
        List<SysAiMessage> all = messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, convId)
                .orderByAsc(SysAiMessage::getId));
        Map<Long, SysAiMessage> byId =
                all.stream().collect(Collectors.toMap(SysAiMessage::getId, m -> m, (a, b) -> a, LinkedHashMap::new));
        List<SysAiMessage> chain = new ArrayList<>();
        Set<Long> visited = new HashSet<>();
        Long cursor = tailMsgId;
        while (cursor != null && visited.add(cursor)) {
            SysAiMessage msg = byId.get(cursor);
            if (msg == null) {
                break;
            }
            chain.add(msg);
            cursor = msg.getParentMessageId();
        }
        Collections.reverse(chain);
        return chain;
    }

    /**
     * 批量装配消息的产物引用：消息不存产物正文，只按 messageId 取 sys_ai_artifact 的 id 列表
     */
    private Map<Long, List<Long>> artifactIdsByMessage(List<Long> messageIds) {
        if (messageIds.isEmpty()) {
            return Map.of();
        }
        Map<Long, List<Long>> grouped = new LinkedHashMap<>();
        for (SysAiArtifactMapper.ArtifactRefRow row : artifactMapper.listRefsByMessageIds(messageIds)) {
            grouped.computeIfAbsent(row.getMessageId(), k -> new ArrayList<>()).add(row.getId());
        }
        return grouped;
    }

    /** 消息正文由 text 块派生（thinking/tool_use 块不进正文，同 kind 按 seq 保序） */
    private static String textOf(SysAiMessage msg) {
        if (msg.getBlocks() == null) {
            return "";
        }
        return msg.getBlocks().stream()
                .filter(block -> "text".equals(block.kind()))
                .map(AiContract.Block::text)
                .filter(text -> text != null && !text.isEmpty())
                .collect(Collectors.joining("\n\n"));
    }

    /**
     * 待恢复中断的读时归一化（列表 / 详情 / 专用端点共用）。
     *
     * <p>只有"消息仍处于 awaiting_input 且未过 expiresAt"才保留可续；其余情况一律
     * {@code actionable=false}，避免历史残留或已过期中断在前端表现为可点击。payload 全部字段
     * 原样保留，只改 {@code actionable}，且不回写库（库内记录是中断发生时刻的事实）。
     */
    private static AiContract.Interrupt normalizePendingInterrupt(SysAiMessage msg) {
        AiContract.Interrupt interrupt = msg.getPendingInterrupt();
        if (interrupt == null) {
            return null;
        }
        boolean actionable = interrupt.actionable()
                && STATUS_AWAITING_INPUT.equals(msg.getStatus())
                && !isExpired(interrupt.expiresAt());
        if (actionable == interrupt.actionable()) {
            return interrupt;
        }
        return new AiContract.Interrupt(
                interrupt.id(),
                interrupt.type(),
                interrupt.confirmKind(),
                interrupt.action(),
                interrupt.recommendation(),
                interrupt.alternatives(),
                interrupt.upgradeTip(),
                interrupt.usedDaily(),
                interrupt.dailyLimit(),
                interrupt.usedMonthly(),
                interrupt.monthlyLimit(),
                interrupt.estDuration(),
                interrupt.imageCount(),
                interrupt.plan(),
                interrupt.reason(),
                interrupt.detail(),
                interrupt.createdAt(),
                interrupt.expiresAt(),
                actionable);
    }

    /**
     * {@code expiresAt} 过期判定：支持带时区偏移的 ISO-8601，也接受库内
     * {@code yyyy-MM-dd HH:mm:ss} 口径。
     *
     * <p>缺失或无法解析按"已过期"处理（对齐 Go {@code NormalizeInterrupt}）：无法确认窗口
     * 就不应给出可续信号，resume 侧也会拒绝。
     */
    private static boolean isExpired(String expiresAt) {
        if (expiresAt == null || expiresAt.isBlank()) {
            return true;
        }
        try {
            return !Instant.now().isBefore(OffsetDateTime.parse(expiresAt).toInstant());
        } catch (DateTimeParseException ignored) {
            // 继续按无时区本地时间解析
        }
        LocalDateTime local = AiDateTimeUtils.parseOrNull(expiresAt);
        return local == null || !LocalDateTime.now().isBefore(local);
    }

    /** LIKE 模式：用户输入先转义 \ % _，再包上前后通配符（配合 SQL 的 {@code ESCAPE '\\'}） */
    private static String likePattern(String keyword) {
        return "%" + escapeLike(keyword) + "%";
    }

    /** LIKE 转义：反斜杠须先替换，避免把后插入的转义符再转义一遍（对齐 python escape_like / go escapeLike） */
    static String escapeLike(String keyword) {
        if (keyword == null || keyword.isEmpty()) {
            return "";
        }
        return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_");
    }

    /**
     * 管理端审计视角补充用户名、消耗汇总与异常标注（批量查询，避免逐会话 N+1）
     */
    private void attachAuditFields(List<AiConversationVO> results) {
        if (results.isEmpty()) {
            return;
        }
        List<Long> convIds = results.stream().map(AiConversationVO::getId).toList();
        Set<Long> userIds = results.stream().map(AiConversationVO::getUserId).collect(Collectors.toSet());
        Map<Long, String> names = insightMapper.listUserDisplayNames(new ArrayList<>(userIds)).stream()
                .collect(Collectors.toMap(
                        DisplayNameRead::getId, r -> r.getName() == null ? "" : r.getName(), (a, b) -> a));
        Map<Long, ConversationConsumptionRead> consumption =
                insightMapper.sumConsumptionByConversationIds(convIds).stream()
                        .collect(Collectors.toMap(ConversationConsumptionRead::getConversationId, r -> r, (a, b) -> a));
        Map<Long, Set<String>> anomalyStatus = new HashMap<>();
        for (AnomalyStatusRead row : insightMapper.listAnomalyStatusByConversations(convIds)) {
            anomalyStatus
                    .computeIfAbsent(row.getConversationId(), k -> new HashSet<>())
                    .add(row.getStatus());
        }
        Set<Long> quotaConvIds = new java.util.HashSet<>(insightMapper.listQuotaAnomalyConversationIds(convIds));
        Set<Long> riskyToolConvIds = new java.util.HashSet<>(insightMapper.listRiskyToolConversationIds(convIds));
        for (AiConversationVO result : results) {
            result.setUserName(names.get(result.getUserId()));
            ConversationConsumptionRead stat = consumption.get(result.getId());
            result.setTokenConsumed(stat == null || stat.getToken() == null ? 0L : stat.getToken());
            result.setCreditsConsumed(stat == null || stat.getCredits() == null ? 0L : stat.getCredits());
            Set<String> statuses = anomalyStatus.getOrDefault(result.getId(), Set.of());
            if (statuses.contains("failed")) {
                result.setAnomalyType("failed");
                result.setAnomalyLabel("存在失败消息");
            } else if (quotaConvIds.contains(result.getId())) {
                result.setAnomalyType("quota");
                result.setAnomalyLabel("配额不足中断");
            } else if (riskyToolConvIds.contains(result.getId())) {
                result.setAnomalyType("risky_tool");
                result.setAnomalyLabel("存在高风险工具调用");
            } else if (statuses.contains("canceled")) {
                result.setAnomalyType("canceled");
                result.setAnomalyLabel("存在已取消消息");
            }
        }
    }

    /**
     * 搜索命中消息内容的消息 ID 回填（标题已命中无需定位）；无 ES 以 DB LIKE 等价定位最新命中消息
     */
    private void attachMatchedMessages(List<AiConversationVO> results, String keyword) {
        List<Long> convIds = results.stream()
                .filter(r -> r.getTitle() == null || !r.getTitle().contains(keyword))
                .map(AiConversationVO::getId)
                .toList();
        if (convIds.isEmpty()) {
            return;
        }
        Map<Long, Long> matched = messageMapper.findLatestIdsByKeyword(convIds, escapeLike(keyword)).stream()
                .collect(Collectors.toMap(
                        SysAiMessageMapper.KeywordMatchRow::getConversationId,
                        SysAiMessageMapper.KeywordMatchRow::getMessageId,
                        (a, b) -> a));
        for (AiConversationVO result : results) {
            result.setMatchedMessageId(matched.get(result.getId()));
        }
    }

    private AiConversationVO toVO(SysAiConversation conv) {
        AiConversationVO vo = new AiConversationVO();
        vo.setId(conv.getId());
        vo.setUserId(conv.getUserId());
        vo.setTitle(conv.getTitle());
        vo.setModel(conv.getModel());
        vo.setAgentCode(conv.getAgentCode());
        vo.setAgentVersion(conv.getAgentVersion());
        vo.setSummary(conv.getSummary());
        vo.setSystemPrompt(conv.getSystemPrompt());
        vo.setModelConfig(conv.getModelConfig());
        vo.setSuggestionsEnabled(conv.getSuggestionsEnabled());
        vo.setApiKeyId(conv.getApiKeyId());
        vo.setMessageCount(conv.getMessageCount());
        vo.setLastMessageAt(conv.getLastMessageAt());
        vo.setCurrentBranchMessageId(conv.getCurrentBranchMessageId());
        vo.setLastReadMessageId(conv.getLastReadMessageId());
        vo.setPinned(conv.getPinned());
        vo.setPinnedAt(conv.getPinnedAt());
        vo.setDeleteTime(conv.getDeleteTime());
        vo.setUnreadCount(0);
        vo.setTitleSource(conv.getTitleSource());
        vo.setStatus(conv.getStatus());
        vo.setCreateTime(conv.getCreateTime());
        vo.setUpdateTime(conv.getUpdateTime());
        return vo;
    }

    private AiConversationVO toVOWithUnread(SysAiConversation conv) {
        AiConversationVO vo = toVO(conv);
        if (conv.getLastReadMessageId() != null) {
            vo.setUnreadCount((int) messageMapper.countMessagesAfter(conv.getId(), conv.getLastReadMessageId()));
        } else {
            vo.setUnreadCount(conv.getMessageCount() == null ? 0 : conv.getMessageCount());
        }
        return vo;
    }

    private AiMessageVO toMessageVO(SysAiMessage msg, List<Long> artifactIds) {
        return toMessageVO(msg, artifactIds, null, null);
    }

    private AiMessageVO toMessageVO(
            SysAiMessage msg, List<Long> artifactIds, Integer siblingIndex, Integer siblingCount) {
        AiMessageVO vo = new AiMessageVO();
        vo.setId(msg.getId());
        vo.setConversationId(msg.getConversationId());
        vo.setParentMessageId(msg.getParentMessageId());
        vo.setRole(msg.getRole());
        vo.setStatus(msg.getStatus());
        vo.setModel(msg.getModel());
        vo.setError(msg.getError());
        vo.setBlocks(msg.getBlocks());
        vo.setPlan(msg.getPlan());
        vo.setPendingInterrupt(normalizePendingInterrupt(msg));
        vo.setUsage(toUsage(msg));
        vo.setArtifactIds(artifactIds);
        vo.setMemoryIds(msg.getUsedMemoryIds() == null ? List.of() : msg.getUsedMemoryIds());
        vo.setSuggestions(msg.getSuggestions() == null ? List.of() : msg.getSuggestions());
        vo.setEdited(Integer.valueOf(1).equals(msg.getEdited()));
        vo.setSiblingIndex(siblingIndex);
        vo.setSiblingCount(siblingCount);
        vo.setCreateTime(msg.getCreateTime());
        vo.setUpdateTime(msg.getUpdateTime());
        return vo;
    }

    private AiContract.Usage toUsage(SysAiMessage msg) {
        return new AiContract.Usage(
                msg.getInputTokens() == null ? 0L : msg.getInputTokens(),
                msg.getOutputTokens() == null ? 0L : msg.getOutputTokens(),
                msg.getCachedInputTokens() == null ? 0L : msg.getCachedInputTokens(),
                msg.getCredits() == null ? 0L : msg.getCredits(),
                msg.getSubAgentUsage());
    }

    private <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(records);
        return result;
    }
}
