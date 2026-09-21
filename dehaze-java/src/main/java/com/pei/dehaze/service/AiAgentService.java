package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalDatasetMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalRunMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalSampleMapper;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiAgentMcpMapper;
import com.pei.dehaze.mapper.SysAiAgentSkillMapper;
import com.pei.dehaze.mapper.SysAiAgentSubagentMapper;
import com.pei.dehaze.mapper.SysAiAgentVersionMapper;
import com.pei.dehaze.mapper.SysAiConversationMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentEvalDataset;
import com.pei.dehaze.model.entity.SysAiAgentMcp;
import com.pei.dehaze.model.entity.SysAiAgentSkill;
import com.pei.dehaze.model.entity.SysAiAgentSubagent;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import com.pei.dehaze.model.form.AiAgentCreateForm;
import com.pei.dehaze.model.form.AiAgentSubAgentsForm;
import com.pei.dehaze.model.form.AiAgentUpdateForm;
import com.pei.dehaze.model.query.AiAgentPageQuery;
import com.pei.dehaze.model.read.AgentRefCountRead;
import com.pei.dehaze.model.vo.AiAgentVO;
import com.pei.dehaze.model.vo.AiSubAgentVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * 智能体管理服务（CRUD/启停/复制/关联绑定/删除级联/缓存失效）。
 *
 * <p>对齐 dehaze-python {@code ai_agent_service}：软删方案 A、关联覆盖式更新与引用完整性校验、
 * 默认 Agent 保护、删除时级联清理评测资产、写操作按 python 键规范失效 Redis 缓存。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiAgentService {

    /** 默认 Agent 编码（系统预置且不可删除） */
    public static final String DEFAULT_AGENT_CODE = "default";

    private final SysAiAgentMapper agentMapper;

    private final SysAiAgentVersionMapper agentVersionMapper;

    private final SysAiAgentSkillMapper agentSkillMapper;

    private final SysAiAgentMcpMapper agentMcpMapper;

    private final SysAiAgentSubagentMapper agentSubagentMapper;

    private final SysAiAgentEvalDatasetMapper evalDatasetMapper;

    private final SysAiAgentEvalSampleMapper evalSampleMapper;

    private final SysAiAgentEvalRunMapper evalRunMapper;

    private final SysAiConversationMapper conversationMapper;

    private final AiInsightMapper insightMapper;

    private final AiCacheInvalidator cacheInvalidator;

    private final AuditLogService auditLogService;

    public IPage<AiAgentVO> list(AiAgentPageQuery query) {
        LambdaQueryWrapper<SysAiAgent> wrapper = new LambdaQueryWrapper<SysAiAgent>()
                .eq(query.getStatus() != null, SysAiAgent::getStatus, query.getStatus())
                .orderByAsc(SysAiAgent::getSortOrder)
                .orderByAsc(SysAiAgent::getId);
        if (query.getKeyword() != null && !query.getKeyword().isBlank()) {
            String keyword = query.getKeyword();
            wrapper.and(w -> w.like(SysAiAgent::getName, keyword).or().like(SysAiAgent::getAgentCode, keyword));
        }
        switch (query.getType() == null ? "" : query.getType()) {
            case "agent" -> wrapper.eq(SysAiAgent::getIsSubagent, 0).eq(SysAiAgent::getIsTeam, 0);
            case "subagent" -> wrapper.eq(SysAiAgent::getIsSubagent, 1);
            case "team" -> wrapper.eq(SysAiAgent::getIsTeam, 1);
            default -> {
            }
        }
        Page<SysAiAgent> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiAgent> agentPage = agentMapper.selectPage(page, wrapper);
        List<Long> agentIds = agentPage.getRecords().stream().map(SysAiAgent::getId).toList();
        Map<Long, Long> skillCounts = countMap(agentIds, insightMapper::countSkillsByAgentIds);
        Map<Long, Long> mcpCounts = countMap(agentIds, insightMapper::countMcpByAgentIds);
        Map<Long, Long> subCounts = countMap(agentIds, insightMapper::countSubagentsByAgentIds);
        List<AiAgentVO> records = new ArrayList<>();
        for (SysAiAgent agent : agentPage.getRecords()) {
            AiAgentVO vo = toVO(agent);
            vo.setSkillCount(skillCounts.getOrDefault(agent.getId(), 0L).intValue());
            vo.setMcpCount(mcpCounts.getOrDefault(agent.getId(), 0L).intValue());
            vo.setSubAgentCount(subCounts.getOrDefault(agent.getId(), 0L).intValue());
            records.add(vo);
        }
        return pageOf(page, records, agentPage.getTotal());
    }

    /**
     * 可选 Agent 列表（启用且非子 Agent；Team 可作会话入口，保留）
     */
    public List<AiAgentVO> listEnabled() {
        return agentMapper.selectList(new LambdaQueryWrapper<SysAiAgent>()
                        .eq(SysAiAgent::getStatus, 1)
                        .eq(SysAiAgent::getIsSubagent, 0)
                        .orderByAsc(SysAiAgent::getSortOrder)
                        .orderByAsc(SysAiAgent::getId))
                .stream().map(this::toVO).toList();
    }

    public AiAgentVO getDetail(Long agentId) {
        return buildDetail(getOrThrow(agentId));
    }

    @Transactional
    public AiAgentVO create(AiAgentCreateForm form) {
        if (findByCode(form.getAgentCode()) != null) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "Agent 编码已存在");
        }
        SysAiAgent agent = new SysAiAgent();
        agent.setAgentCode(form.getAgentCode());
        agent.setName(form.getName());
        agent.setDescription(form.getDescription() == null ? "" : form.getDescription());
        agent.setSystemPrompt(form.getSystemPrompt());
        agent.setModelId(form.getModelId());
        // 推理范式：python 侧为 pattern ^(auto|direct|react|plan_execute|reflexion)$，属非连续多值枚举，
        // 按裁决走 service 层白名单（不搬 @Pattern），越界值原会原样落库成脏数据
        String reasoningMode = form.getReasoningMode() == null ? "auto" : form.getReasoningMode();
        requireReasoningMode(reasoningMode);
        agent.setReasoningMode(reasoningMode);
        agent.setConfig(form.getConfig());
        agent.setIsSubagent(flag(form.getIsSubagent()));
        agent.setIsTeam(flag(form.getIsTeam()));
        agent.setIsExposed(flag(form.getIsExposed()));
        agent.setPermissions(form.getPermissions());
        agent.setTags(form.getTags());
        agent.setSortOrder(form.getSortOrder() == null ? 0 : form.getSortOrder());
        agent.setStatus(form.getStatus() == null ? 1 : form.getStatus());
        agentMapper.insert(agent);
        cacheInvalidator.evict("ai:agent:list:enabled");
        return buildDetail(agent);
    }

    /** 推理范式白名单（与 python AgentCreate/AgentUpdate.reasoning_mode 的 pattern 同集合） */
    private void requireReasoningMode(String mode) {
        if (!java.util.Set.of("auto", "direct", "react", "plan_execute", "reflexion").contains(mode)) {
            throw new BusinessException(ResultCode.PARAM_ERROR,
                    "推理范式仅支持auto/direct/react/plan_execute/reflexion");
        }
    }

    @Transactional
    public AiAgentVO update(Long agentId, AiAgentUpdateForm form) {
        SysAiAgent agent = getOrThrow(agentId);
        if (form.getName() != null) {
            agent.setName(form.getName());
        }
        if (form.getDescription() != null) {
            agent.setDescription(form.getDescription());
        }
        if (form.getSystemPrompt() != null) {
            agent.setSystemPrompt(form.getSystemPrompt());
        }
        if (form.getModelId() != null) {
            agent.setModelId(form.getModelId());
        }
        if (form.getReasoningMode() != null) {
            requireReasoningMode(form.getReasoningMode());
            agent.setReasoningMode(form.getReasoningMode());
        }
        if (form.getConfig() != null) {
            agent.setConfig(form.getConfig());
        }
        if (form.getIsSubagent() != null) {
            agent.setIsSubagent(flag(form.getIsSubagent()));
        }
        if (form.getIsTeam() != null) {
            agent.setIsTeam(flag(form.getIsTeam()));
        }
        if (form.getIsExposed() != null) {
            agent.setIsExposed(flag(form.getIsExposed()));
        }
        if (form.getPermissions() != null) {
            agent.setPermissions(form.getPermissions());
        }
        if (form.getTags() != null) {
            agent.setTags(form.getTags());
        }
        if (form.getSortOrder() != null) {
            agent.setSortOrder(form.getSortOrder());
        }
        agentMapper.updateById(agent);
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agent.getId());
        return buildDetail(agent);
    }

    @Transactional
    public void setStatus(Long agentId, Integer status) {
        SysAiAgent agent = getOrThrow(agentId);
        agent.setStatus(status);
        agentMapper.updateById(agent);
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agent.getId());
    }

    /**
     * 删除 Agent：默认 Agent 保护 → 会话/子 Agent 引用校验 → 级联清理评测资产 → 软删 → 审计
     *
     * <p>评测资产按 agent_id 挂载，Agent 软删后即失去清理入口；agent_id 不复用，重建同编码得到新 id，
     * 故删除时一并清理：评测集软删、样本与执行记录物理删除（两表无逻辑删除列）。
     */
    @Transactional
    public void delete(Long agentId) {
        SysAiAgent agent = getOrThrow(agentId);
        if (DEFAULT_AGENT_CODE.equals(agent.getAgentCode())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "默认 Agent 不可删除");
        }
        long conversationRefs = conversationMapper.countByAgentCode(agent.getAgentCode());
        if (conversationRefs > 0) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                    "存在 " + conversationRefs + " 个会话正在使用该 Agent，请先解绑");
        }
        long subagentRefs = insightMapper.countSubagentReferences(agentId);
        if (subagentRefs > 0) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                    "该 Agent 被 " + subagentRefs + " 个 Agent 作为子 Agent 引用，请先解绑");
        }
        List<SysAiAgentEvalDataset> datasets = evalDatasetMapper.selectList(
                new LambdaQueryWrapper<SysAiAgentEvalDataset>().eq(SysAiAgentEvalDataset::getAgentId, agentId));
        List<Long> datasetIds = datasets.stream().map(SysAiAgentEvalDataset::getId).toList();
        int sampleCount = datasetIds.isEmpty() ? 0 : evalSampleMapper.deleteByDatasetIds(datasetIds);
        if (!datasetIds.isEmpty()) {
            evalDatasetMapper.softDeleteByIds(datasetIds);
        }
        int runCount = evalRunMapper.deleteByAgentId(agentId);
        agentMapper.deleteById(agentId);
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agentId);
        Map<String, Object> beforeValue = new LinkedHashMap<>();
        beforeValue.put("agent_code", agent.getAgentCode());
        beforeValue.put("name", agent.getName());
        Map<String, Object> afterValue = new LinkedHashMap<>();
        afterValue.put("eval_datasets_soft_deleted", datasetIds.size());
        afterValue.put("eval_samples_deleted", sampleCount);
        afterValue.put("eval_runs_deleted", runCount);
        Long operatorId = com.pei.dehaze.security.util.SecurityUtils.getUserId();
        AiTxSupport.afterCommit(() -> auditLogService.recordAudit(operatorId, "ai_agent", agentId, "delete",
                "ai_agent", beforeValue, afterValue, null, null));
    }

    @Transactional
    public AiAgentVO copy(Long agentId, String newAgentCode) {
        SysAiAgent source = getOrThrow(agentId);
        if (findByCode(newAgentCode) != null) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "Agent 编码已存在");
        }
        SysAiAgent copy = new SysAiAgent();
        copy.setAgentCode(newAgentCode);
        copy.setName(source.getName());
        copy.setDescription(source.getDescription());
        copy.setSystemPrompt(source.getSystemPrompt());
        copy.setModelId(source.getModelId());
        copy.setReasoningMode(source.getReasoningMode());
        copy.setConfig(source.getConfig());
        copy.setIsSubagent(source.getIsSubagent());
        copy.setIsTeam(source.getIsTeam());
        copy.setIsExposed(source.getIsExposed());
        copy.setPermissions(source.getPermissions());
        copy.setTags(source.getTags());
        copy.setSortOrder(source.getSortOrder());
        copy.setStatus(1);
        agentMapper.insert(copy);
        cacheInvalidator.evict("ai:agent:list:enabled");
        return buildDetail(copy);
    }

    /**
     * 设置 Skills（覆盖式）：引用完整性校验，缺失名称一次性报错
     */
    @Transactional
    public void setSkills(Long agentId, List<String> skillNames) {
        SysAiAgent agent = getOrThrow(agentId);
        List<String> names = skillNames == null ? List.of() : skillNames;
        if (!names.isEmpty()) {
            Set<String> existing = new HashSet<>(agentSkillMapper.listExistingSkillNames(names));
            List<String> missing = names.stream().filter(n -> !existing.contains(n)).sorted().toList();
            if (!missing.isEmpty()) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND,
                        "以下 Skill 不存在: " + joinTop5(missing));
            }
        }
        agentSkillMapper.delete(new LambdaQueryWrapper<SysAiAgentSkill>()
                .eq(SysAiAgentSkill::getAgentId, agentId));
        for (String name : names) {
            SysAiAgentSkill link = new SysAiAgentSkill();
            link.setAgentId(agentId);
            link.setSkillName(name);
            link.setCreateTime(LocalDateTime.now());
            agentSkillMapper.insert(link);
        }
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agentId);
    }

    /**
     * 设置 MCP 命名空间（覆盖式）：命名空间必须已在注册 MCP Server 下声明，否则运行时装载不到工具
     */
    @Transactional
    public void setMcpNamespaces(Long agentId, List<String> namespaces) {
        SysAiAgent agent = getOrThrow(agentId);
        List<String> names = namespaces == null ? List.of() : namespaces;
        if (!names.isEmpty()) {
            Set<String> registered = new HashSet<>(agentMcpMapper.listRegisteredNamespaces(names));
            List<String> missing = names.stream().filter(n -> !registered.contains(n)).sorted().toList();
            if (!missing.isEmpty()) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND,
                        "以下 MCP 命名空间未注册: " + joinTop5(missing));
            }
        }
        agentMcpMapper.delete(new LambdaQueryWrapper<SysAiAgentMcp>().eq(SysAiAgentMcp::getAgentId, agentId));
        for (String namespace : names) {
            SysAiAgentMcp link = new SysAiAgentMcp();
            link.setAgentId(agentId);
            link.setMcpNamespace(namespace);
            link.setCreateTime(LocalDateTime.now());
            agentMcpMapper.insert(link);
        }
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agentId);
    }

    /**
     * 设置子 Agent（覆盖式）：自引用校验 → 存在性校验 → 环检测（环会让推理期子代理展开无限递归）
     */
    @Transactional
    public void setSubagents(Long agentId, AiAgentSubAgentsForm form) {
        SysAiAgent agent = getOrThrow(agentId);
        List<com.pei.dehaze.model.form.AiSubAgentItemForm> items =
                form.getSubagents() == null ? List.of() : form.getSubagents();
        List<Long> childIds = items.stream().map(com.pei.dehaze.model.form.AiSubAgentItemForm::getAgentId).toList();
        if (childIds.contains(agentId)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "子 Agent 不能是自身");
        }
        if (!childIds.isEmpty()) {
            Set<Long> found = agentMapper.selectList(new LambdaQueryWrapper<SysAiAgent>()
                            .select(SysAiAgent::getId).in(SysAiAgent::getId, childIds))
                    .stream().map(SysAiAgent::getId).collect(java.util.stream.Collectors.toSet());
            List<Long> missing = childIds.stream().filter(id -> !found.contains(id)).distinct().sorted().toList();
            if (!missing.isEmpty()) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND,
                        "以下子 Agent 不存在: " + joinTop5(missing.stream().map(String::valueOf).toList()));
            }
        }
        ensureSubagentsAcyclic(agentId, childIds);
        agentSubagentMapper.delete(new LambdaQueryWrapper<SysAiAgentSubagent>()
                .eq(SysAiAgentSubagent::getParentAgentId, agentId));
        for (com.pei.dehaze.model.form.AiSubAgentItemForm item : items) {
            SysAiAgentSubagent link = new SysAiAgentSubagent();
            link.setParentAgentId(agentId);
            link.setSubagentAgentId(item.getAgentId());
            link.setEndpointId(item.getEndpointId());
            link.setPriority(item.getPriority() == null ? 0 : item.getPriority());
            link.setCreateTime(LocalDateTime.now());
            agentSubagentMapper.insert(link);
        }
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agentId);
    }

    /**
     * 子 Agent 环检测（DFS）：绑定后沿子关系展开，命中当前路径上的节点即为环
     */
    private void ensureSubagentsAcyclic(Long agentId, List<Long> childIds) {
        List<Long> path = new ArrayList<>();
        Set<Long> settled = new HashSet<>();
        walk(agentId, childIds, path, settled);
    }

    private void walk(Long node, List<Long> rootChildren, List<Long> path, Set<Long> settled) {
        if (settled.contains(node)) {
            return;
        }
        if (path.contains(node)) {
            List<Long> cyclePath = new ArrayList<>(path.subList(path.indexOf(node), path.size()));
            cyclePath.add(node);
            throw new BusinessException(ResultCode.PARAM_ERROR,
                    "子 Agent 绑定存在环: " + cyclePath.stream().map(String::valueOf)
                            .collect(java.util.stream.Collectors.joining("→")));
        }
        path.add(node);
        List<Long> children = node.equals(path.get(0)) ? rootChildren : listSubagentIds(node);
        for (Long child : children) {
            walk(child, rootChildren, path, settled);
        }
        path.remove(path.size() - 1);
        settled.add(node);
    }

    private List<Long> listSubagentIds(Long parentAgentId) {
        return agentSubagentMapper.selectList(new LambdaQueryWrapper<SysAiAgentSubagent>()
                        .eq(SysAiAgentSubagent::getParentAgentId, parentAgentId))
                .stream().map(SysAiAgentSubagent::getSubagentAgentId).toList();
    }

    // ── 内部工具 ─────────────────────────────────────────────

    public SysAiAgent getOrThrow(Long agentId) {
        SysAiAgent agent = agentMapper.selectById(agentId);
        if (agent == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在");
        }
        return agent;
    }

    /**
     * 主表可编辑态回写（版本回滚链路使用）
     */
    public void updateById(SysAiAgent agent) {
        agentMapper.updateById(agent);
    }

    private SysAiAgent findByCode(String agentCode) {
        return agentMapper.selectOne(new LambdaQueryWrapper<SysAiAgent>()
                .eq(SysAiAgent::getAgentCode, agentCode).last("LIMIT 1"));
    }

    private Integer flag(Boolean value) {
        return Boolean.TRUE.equals(value) ? 1 : 0;
    }

    private String joinTop5(List<String> values) {
        return String.join(", ", values.subList(0, Math.min(5, values.size())));
    }

    private Map<Long, Long> countMap(List<Long> agentIds, Function<List<Long>, List<AgentRefCountRead>> loader) {
        if (agentIds.isEmpty()) {
            return Map.of();
        }
        Map<Long, Long> result = new HashMap<>();
        for (AgentRefCountRead row : loader.apply(agentIds)) {
            result.put(row.getAgentId(), row.getCnt());
        }
        return result;
    }

    /**
     * 详情：基本信息 + 关联 Skill/命名空间/子 Agent（子 Agent 含名称与编码，供前端展示触发描述）
     */
    public AiAgentVO buildDetail(SysAiAgent agent) {
        AiAgentVO vo = toVO(agent);
        List<String> skills = agentSkillMapper.selectList(new LambdaQueryWrapper<SysAiAgentSkill>()
                        .eq(SysAiAgentSkill::getAgentId, agent.getId()))
                .stream().map(SysAiAgentSkill::getSkillName).toList();
        List<String> namespaces = agentMcpMapper.selectList(new LambdaQueryWrapper<SysAiAgentMcp>()
                        .eq(SysAiAgentMcp::getAgentId, agent.getId()))
                .stream().map(SysAiAgentMcp::getMcpNamespace).toList();
        List<SysAiAgentSubagent> links = agentSubagentMapper.selectList(
                new LambdaQueryWrapper<SysAiAgentSubagent>().eq(SysAiAgentSubagent::getParentAgentId, agent.getId()));
        Map<Long, SysAiAgent> subAgents = new HashMap<>();
        if (!links.isEmpty()) {
            List<Long> subIds = links.stream().map(SysAiAgentSubagent::getSubagentAgentId).toList();
            for (SysAiAgent sub : agentMapper.selectList(new LambdaQueryWrapper<SysAiAgent>()
                    .in(SysAiAgent::getId, subIds))) {
                subAgents.put(sub.getId(), sub);
            }
        }
        List<AiSubAgentVO> subagents = new ArrayList<>();
        for (SysAiAgentSubagent link : links) {
            SysAiAgent sub = subAgents.get(link.getSubagentAgentId());
            AiSubAgentVO item = new AiSubAgentVO();
            item.setAgentId(link.getSubagentAgentId());
            item.setAgentName(sub == null ? "" : sub.getName());
            item.setAgentCode(sub == null ? "" : sub.getAgentCode());
            item.setDescription(sub == null ? "" : sub.getDescription());
            item.setEndpointId(link.getEndpointId());
            item.setPriority(link.getPriority());
            subagents.add(item);
        }
        vo.setSystemPrompt(agent.getSystemPrompt());
        vo.setConfig(agent.getConfig());
        vo.setPermissions(agent.getPermissions());
        vo.setSkills(skills);
        vo.setMcpNamespaces(namespaces);
        vo.setSubagents(subagents);
        vo.setSkillCount(skills.size());
        vo.setMcpCount(namespaces.size());
        vo.setSubAgentCount(subagents.size());
        return vo;
    }

    private AiAgentVO toVO(SysAiAgent agent) {
        AiAgentVO vo = new AiAgentVO();
        vo.setId(agent.getId());
        vo.setAgentCode(agent.getAgentCode());
        vo.setName(agent.getName());
        vo.setDescription(agent.getDescription());
        vo.setModelId(agent.getModelId());
        vo.setReasoningMode(agent.getReasoningMode());
        vo.setIsSubagent(agent.getIsSubagent());
        vo.setIsTeam(agent.getIsTeam());
        vo.setIsExposed(agent.getIsExposed());
        vo.setTags(agent.getTags() == null ? List.of() : agent.getTags());
        vo.setStatus(agent.getStatus());
        vo.setSortOrder(agent.getSortOrder());
        vo.setCreateTime(agent.getCreateTime());
        return vo;
    }

    private <T> IPage<T> pageOf(Page<?> page, List<T> records, long total) {
        Page<T> result = new Page<>(page.getCurrent(), page.getSize(), total);
        result.setRecords(new ArrayList<>(records));
        return result;
    }
}
