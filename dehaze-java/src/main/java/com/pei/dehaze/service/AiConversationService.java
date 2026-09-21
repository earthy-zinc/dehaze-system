package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
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
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentThought;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import com.pei.dehaze.model.entity.SysAiConversation;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.form.AiConversationBatchForm;
import com.pei.dehaze.model.form.AiConversationCreateForm;
import com.pei.dehaze.model.form.AiConversationUpdateForm;
import com.pei.dehaze.model.query.AiConversationPageQuery;
import com.pei.dehaze.model.read.AnomalyStatusRead;
import com.pei.dehaze.model.read.ConversationConsumptionRead;
import com.pei.dehaze.model.read.DisplayNameRead;
import com.pei.dehaze.model.vo.AiAgentThoughtVO;
import com.pei.dehaze.model.vo.AiConversationVO;
import com.pei.dehaze.model.vo.AiMessagePageVO;
import com.pei.dehaze.model.vo.AiMessageVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDateTime;
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

/**
 * AI 会话与消息服务（非推理类端点）。
 *
 * <p>行为对齐 dehaze-python {@code ai_conversation_service}：软删方案 A（deleted=id + delete_time）、
 * 回收站 30 天窗口、置顶上限与用户级锁、消息分页取最新一页。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiConversationService {

    /** 未指定 Agent 时的平台默认 Agent 编码 */
    private static final String DEFAULT_AGENT_CODE = "default";

    /** 会话默认模型标识（对齐 python AI_DEFAULT_MODEL） */
    private static final String DEFAULT_MODEL = "qwen3-0.6b";

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

    private final SysAiAgentThoughtMapper thoughtMapper;

    private final SysAiAgentMapper agentMapper;

    private final SysAiAgentVersionMapper agentVersionMapper;

    private final AiInsightMapper insightMapper;

    private final StringRedisTemplate stringRedisTemplate;

    private final ObjectMapper objectMapper;

    /**
     * 会话导出结果（控制器直接写响应体）
     */
    public record ConversationExport(String filename, String contentType, String content) {
    }

    // ── 会话 CRUD ─────────────────────────────────────────────

    @Transactional
    public AiConversationVO create(Long userId, AiConversationCreateForm form) {
        String scene = form.getScene() != null && SCENE_PROMPTS.containsKey(form.getScene())
                ? form.getScene() : "general";
        AgentAnchor anchor = resolveAgentAnchor(form.getAgentCode());
        SysAiConversation conv = new SysAiConversation();
        conv.setUserId(userId);
        conv.setTitle(form.getTitle() != null && !form.getTitle().isBlank() ? form.getTitle() : "新对话");
        conv.setModel(form.getModel() != null ? form.getModel() : DEFAULT_MODEL);
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
        boolean keywordPresent = query.getKeyword() != null && !query.getKeyword().isBlank();
        LambdaQueryWrapper<SysAiConversation> wrapper = new LambdaQueryWrapper<SysAiConversation>()
                .eq(!admin, SysAiConversation::getUserId, userId)
                .eq(statusFilter != null, SysAiConversation::getStatus, statusFilter)
                .orderByDesc(SysAiConversation::getPinned)
                .orderByDesc(SysAiConversation::getPinnedAt)
                .orderByDesc(SysAiConversation::getLastMessageAt)
                .orderByDesc(SysAiConversation::getId);
        if (keywordPresent) {
            String keyword = query.getKeyword();
            if (admin) {
                wrapper.and(w -> w.like(SysAiConversation::getTitle, keyword));
            } else {
                // 无 ES 全文检索能力，以标题 LIKE + 消息正文命中等价定位（命中消息 ID 供前端定位）
                List<Long> hitConvIds = messageMapper.listConversationIdsByKeyword(keyword);
                if (hitConvIds.isEmpty()) {
                    wrapper.and(w -> w.like(SysAiConversation::getTitle, keyword));
                } else {
                    wrapper.and(w -> w.like(SysAiConversation::getTitle, keyword)
                            .or().in(SysAiConversation::getId, hitConvIds));
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
        SysAiConversation conv = conversationMapper.selectInTrash(convId, userId,
                LocalDateTime.now().minusDays(RECYCLE_WINDOW_DAYS));
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
        IPage<SysAiConversation> trash = conversationMapper.selectTrashPage(page, userId,
                LocalDateTime.now().minusDays(RECYCLE_WINDOW_DAYS));
        List<AiConversationVO> records = trash.getRecords().stream().map(this::toVOWithUnread).toList();
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
                default -> throw new BusinessException(ResultCode.PARAM_ERROR,
                        "不支持的批量操作类型: " + form.getAction());
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
     * 导出会话：沿当前激活分支回溯全部消息，仅导 user/assistant 正文（推理过程与工具调用不导出）
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
            payload.put("conversation", Map.of(
                    "id", conv.getId(),
                    "title", conv.getTitle() == null ? "" : conv.getTitle(),
                    "model", conv.getModel() == null ? "" : conv.getModel(),
                    "agent_code", conv.getAgentCode() == null ? "" : conv.getAgentCode(),
                    "create_time", conv.getCreateTime() == null ? "" : conv.getCreateTime().toString()));
            List<Map<String, Object>> messages = new ArrayList<>();
            for (SysAiMessage msg : chain) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("role", msg.getRole());
                item.put("content", msg.getContent() == null ? "" : msg.getContent());
                item.put("create_time", msg.getCreateTime() == null ? "" : msg.getCreateTime().toString());
                messages.add(item);
            }
            payload.put("messages", messages);
            try {
                return new ConversationExport("conversation_" + convId + ".json", "application/json",
                        objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(payload));
            } catch (Exception e) {
                throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "会话导出失败");
            }
        }
        StringBuilder sb = new StringBuilder("# ").append(conv.getTitle()).append("\n\n");
        for (SysAiMessage msg : chain) {
            sb.append("## ").append("user".equals(msg.getRole()) ? "用户" : "助手").append("\n\n")
                    .append(msg.getContent() == null ? "" : msg.getContent()).append("\n\n");
        }
        return new ConversationExport("conversation_" + convId + ".md", "text/markdown", sb.toString());
    }

    // ── 会话消息 ─────────────────────────────────────────────

    /**
     * 会话消息列表（游标分页：id 倒序，仅取 {@code id < before}；缺省取最新一页）。
     *
     * <p>{@code hasMore} 由"取 limit+1 条"判定——多取的一条即为"是否存在比本页最后一条更早的消息"，
     * 单次查询得出，不额外往返；{@code total} 为会话消息总数（与游标无关）。
     * 关掉 {@code searchCount} 避免 MyBatis-Plus 自动附加的 count 查询。
     */
    public AiMessagePageVO listMessages(Long convId, Long userId, Long before, int limit, boolean admin) {
        SysAiConversation conv = admin ? conversationMapper.selectById(convId) : getOwned(convId, userId);
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
        }
        Page<SysAiMessage> cursor = new Page<>(1, limit + 1L, false);
        List<SysAiMessage> rows = messageMapper.selectPage(cursor,
                new LambdaQueryWrapper<SysAiMessage>()
                        .eq(SysAiMessage::getConversationId, convId)
                        .lt(before != null, SysAiMessage::getId, before)
                        .orderByDesc(SysAiMessage::getId)).getRecords();
        boolean hasMore = rows.size() > limit;
        List<SysAiMessage> pageRecords = hasMore ? rows.subList(0, limit) : rows;
        long total = messageMapper.selectCount(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, convId));
        List<Long> assistantIds = pageRecords.stream()
                .filter(m -> "assistant".equals(m.getRole()))
                .map(SysAiMessage::getId)
                .toList();
        Map<Long, List<SysAiAgentThought>> thoughtsMap = listThoughts(assistantIds);
        List<AiMessageVO> records = new ArrayList<>();
        for (SysAiMessage msg : pageRecords) {
            AiMessageVO vo = toMessageVO(msg);
            vo.setThoughts(thoughtsMap.getOrDefault(msg.getId(), List.of()).stream()
                    .map(this::toThoughtVO).toList());
            records.add(vo);
        }
        AiMessagePageVO result = new AiMessagePageVO();
        result.setList(records);
        result.setTotal(total);
        result.setHasMore(hasMore);
        return result;
    }

    public AiMessageVO getMessage(Long msgId, Long userId, boolean admin) {
        SysAiMessage msg = admin ? messageMapper.selectById(msgId) : getOwnedMessage(msgId, userId);
        if (msg == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        AiMessageVO vo = toMessageVO(msg);
        List<SysAiAgentThought> thoughts = thoughtMapper.selectList(new LambdaQueryWrapper<SysAiAgentThought>()
                .eq(SysAiAgentThought::getMessageId, msgId)
                .orderByAsc(SysAiAgentThought::getPosition));
        vo.setThoughts(thoughts.stream().map(this::toThoughtVO).toList());
        return vo;
    }

    public List<AiMessageVO> getBranches(Long convId, Long userId, Long msgId) {
        getOwned(convId, userId);
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null || !Objects.equals(msg.getConversationId(), convId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        return messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                        .eq(SysAiMessage::getConversationId, convId)
                        .eq(SysAiMessage::getParentMessageId, msgId)
                        .orderByDesc(SysAiMessage::getCreateTime)
                        .orderByDesc(SysAiMessage::getId))
                .stream().map(this::toMessageVO).toList();
    }

    @Transactional
    public AiConversationVO switchBranch(Long convId, Long userId, Long msgId) {
        SysAiConversation conv = getOwned(convId, userId);
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null || !Objects.equals(msg.getConversationId(), convId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        conversationMapper.updateCurrentBranch(convId, msgId);
        conv.setCurrentBranchMessageId(msgId);
        return toVO(conv);
    }

    @Transactional
    public void deleteMessage(Long msgId, Long userId) {
        SysAiMessage msg = getOwnedMessage(msgId, userId);
        if (msg == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        if (!"assistant".equals(msg.getRole())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可删除");
        }
        messageMapper.softDeleteByIds(List.of(msg.getId()));
    }

    // ── 内部工具 ─────────────────────────────────────────────

    /**
     * 会话锚定的 Agent 编码与已发布版本号（无已发布版本时为 null）
     */
    private record AgentAnchor(String code, Integer version) {
    }

    private AgentAnchor resolveAgentAnchor(String agentCode) {
        String code = agentCode == null || agentCode.isBlank() ? DEFAULT_AGENT_CODE : agentCode.trim();
        SysAiAgent agent = agentMapper.selectOne(new LambdaQueryWrapper<SysAiAgent>()
                .eq(SysAiAgent::getAgentCode, code).last("LIMIT 1"));
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
        return conversationMapper.selectList(new LambdaQueryWrapper<SysAiConversation>()
                        .select(SysAiConversation::getId)
                        .eq(SysAiConversation::getUserId, userId)
                        .in(SysAiConversation::getId, convIds))
                .stream().map(SysAiConversation::getId).collect(Collectors.toSet());
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
     * 按当前激活分支末端回溯完整消息链（一次查询本会话消息 + 内存组链，visited 防环），返回正序链
     */
    private List<SysAiMessage> chainByTail(Long convId, Long tailMsgId) {
        List<SysAiMessage> all = messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, convId)
                .orderByAsc(SysAiMessage::getId));
        Map<Long, SysAiMessage> byId = all.stream()
                .collect(Collectors.toMap(SysAiMessage::getId, m -> m, (a, b) -> a, LinkedHashMap::new));
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

    private Map<Long, List<SysAiAgentThought>> listThoughts(List<Long> messageIds) {
        if (messageIds.isEmpty()) {
            return Map.of();
        }
        List<SysAiAgentThought> thoughts = thoughtMapper.selectList(new LambdaQueryWrapper<SysAiAgentThought>()
                .in(SysAiAgentThought::getMessageId, messageIds)
                .orderByAsc(SysAiAgentThought::getPosition));
        Map<Long, List<SysAiAgentThought>> grouped = new HashMap<>();
        for (SysAiAgentThought thought : thoughts) {
            grouped.computeIfAbsent(thought.getMessageId(), k -> new ArrayList<>()).add(thought);
        }
        return grouped;
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
                .collect(Collectors.toMap(DisplayNameRead::getId,
                        r -> r.getName() == null ? "" : r.getName(), (a, b) -> a));
        Map<Long, ConversationConsumptionRead> consumption = insightMapper
                .sumConsumptionByConversationIds(convIds).stream()
                .collect(Collectors.toMap(ConversationConsumptionRead::getConversationId, r -> r, (a, b) -> a));
        Map<Long, Set<Integer>> anomalyStatus = new HashMap<>();
        for (AnomalyStatusRead row : insightMapper.listAnomalyStatusByConversations(convIds)) {
            anomalyStatus.computeIfAbsent(row.getConversationId(), k -> new HashSet<>())
                    .add(row.getStatus());
        }
        Set<Long> quotaConvIds = new java.util.HashSet<>(insightMapper.listQuotaAnomalyConversationIds(convIds));
        Set<Long> riskyToolConvIds = new java.util.HashSet<>(
                insightMapper.listRiskyToolConversationIds(convIds));
        for (AiConversationVO result : results) {
            result.setUserName(names.get(result.getUserId()));
            ConversationConsumptionRead stat = consumption.get(result.getId());
            result.setTokenConsumed(stat == null || stat.getToken() == null ? 0L : stat.getToken());
            result.setCreditsConsumed(stat == null || stat.getCredits() == null ? 0L : stat.getCredits());
            Set<Integer> statuses = anomalyStatus.getOrDefault(result.getId(), Set.of());
            if (statuses.contains(3)) {
                result.setAnomalyType("failed");
                result.setAnomalyLabel("存在失败消息");
            } else if (quotaConvIds.contains(result.getId())) {
                result.setAnomalyType("quota");
                result.setAnomalyLabel("配额不足中断");
            } else if (riskyToolConvIds.contains(result.getId())) {
                result.setAnomalyType("risky_tool");
                result.setAnomalyLabel("存在高风险工具调用");
            } else if (statuses.contains(4)) {
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
        Map<Long, Long> matched = messageMapper.findLatestIdsByKeyword(convIds, keyword).stream()
                .collect(Collectors.toMap(SysAiMessageMapper.KeywordMatchRow::getConversationId,
                        SysAiMessageMapper.KeywordMatchRow::getMessageId, (a, b) -> a));
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

    private AiMessageVO toMessageVO(SysAiMessage msg) {
        AiMessageVO vo = new AiMessageVO();
        vo.setId(msg.getId());
        vo.setConversationId(msg.getConversationId());
        vo.setParentMessageId(msg.getParentMessageId());
        vo.setRole(msg.getRole());
        vo.setContent(msg.getContent());
        vo.setToolCalls(msg.getToolCalls());
        vo.setToolCallId(msg.getToolCallId());
        vo.setModel(msg.getModel());
        vo.setStatus(msg.getStatus());
        vo.setError(msg.getError());
        vo.setMetadata(msg.getMetadata());
        vo.setInputTokens(msg.getInputTokens());
        vo.setOutputTokens(msg.getOutputTokens());
        vo.setCachedInputTokens(msg.getCachedInputTokens());
        vo.setCredits(msg.getCredits());
        vo.setTaskId(msg.getTaskId());
        vo.setUsedMemoryIds(msg.getUsedMemoryIds());
        vo.setEdited(msg.getEdited());
        vo.setOriginalContent(msg.getOriginalContent());
        vo.setCreateTime(msg.getCreateTime());
        return vo;
    }

    private AiAgentThoughtVO toThoughtVO(SysAiAgentThought thought) {
        AiAgentThoughtVO vo = new AiAgentThoughtVO();
        vo.setId(thought.getId());
        vo.setMessageId(thought.getMessageId());
        vo.setConversationId(thought.getConversationId());
        vo.setPosition(thought.getPosition());
        vo.setAgentCode(thought.getAgentCode());
        vo.setIsSubagent(thought.getIsSubagent());
        vo.setThought(thought.getThought());
        vo.setTool(thought.getTool());
        vo.setToolInput(thought.getToolInput());
        vo.setObservation(thought.getObservation());
        vo.setStatus(thought.getStatus());
        vo.setLatencyMs(thought.getLatencyMs());
        vo.setError(thought.getError());
        vo.setCreateTime(thought.getCreateTime());
        return vo;
    }

    private <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(records);
        return result;
    }
}
