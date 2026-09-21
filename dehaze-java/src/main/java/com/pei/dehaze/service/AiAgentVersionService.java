package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiAgentMcpMapper;
import com.pei.dehaze.mapper.SysAiAgentSkillMapper;
import com.pei.dehaze.mapper.SysAiAgentSubagentMapper;
import com.pei.dehaze.mapper.SysAiAgentVersionMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentMcp;
import com.pei.dehaze.model.entity.SysAiAgentSkill;
import com.pei.dehaze.model.entity.SysAiAgentSubagent;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import com.pei.dehaze.model.vo.AiAgentVersionVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * 智能体版本管理服务（草稿快照、发布门禁、回滚、版本历史与差异对比）。
 *
 * <p>对齐 dehaze-python {@code ai_agent_version_service}：快照冻结"继承默认"语义（resolved_config）、
 * 版本号 MAX+1 撞唯一键重试、回滚限定已发布版本且写新版本不覆盖历史。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiAgentVersionService {

    /** 版本状态：2 已发布（草稿态由发布链路在 python 侧产生，java 只读已发布版本） */
    private static final int STATUS_PUBLISHED = 2;

    /** 取号（MAX+1）与插入非原子，并发发布撞唯一键时的重试次数 */
    private static final int VERSION_NO_CONFLICT_RETRY = 5;

    private final SysAiAgentVersionMapper versionMapper;

    private final SysAiAgentSkillMapper skillMapper;

    private final SysAiAgentMcpMapper mcpMapper;

    private final SysAiAgentSubagentMapper subagentMapper;

    private final AiAgentService agentService;

    private final AiAgentConfigResolver configResolver;

    private final AiCacheInvalidator cacheInvalidator;

    private final AuditLogService auditLogService;

    /**
     * 回滚到历史已发布版本：快照覆盖主表可编辑态 + 关联覆盖式恢复 + 写新已发布版本（历史不覆盖）
     */
    @Transactional
    public int rollback(Long agentId, Integer versionNo, Long operatorId) {
        SysAiAgent agent = agentService.getOrThrow(agentId);
        SysAiAgentVersion target = versionMapper.getByAgentAndVersion(agentId, versionNo);
        if (target == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "回滚目标版本不存在");
        }
        if (!Integer.valueOf(STATUS_PUBLISHED).equals(target.getStatus())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅可回滚到已发布版本");
        }
        Map<String, Object> snapshot = target.getSnapshot() == null ? Map.of() : target.getSnapshot();
        agent.setName(stringValue(snapshot, "name", agent.getName()));
        agent.setDescription(stringValue(snapshot, "description", agent.getDescription()));
        agent.setSystemPrompt(snapshot.get("system_prompt") == null ? null
                : String.valueOf(snapshot.get("system_prompt")));
        agent.setModelId(stringValue(snapshot, "model_id", agent.getModelId()));
        agent.setReasoningMode(stringValue(snapshot, "reasoning_mode", agent.getReasoningMode()));
        agent.setConfig(asMap(snapshot.get("config")));
        agent.setPermissions(asMapList(snapshot.get("permissions")));
        agent.setIsSubagent(intValue(snapshot, "is_subagent", agent.getIsSubagent()));
        agent.setIsTeam(intValue(snapshot, "is_team", agent.getIsTeam()));
        agent.setIsExposed(intValue(snapshot, "is_exposed", agent.getIsExposed()));
        agentService.updateById(agent);
        replaceRelations(agentId, snapshot);
        versionMapper.demotePublished(agentId);
        SysAiAgentVersion version = writeVersion(agent, operatorId, "回滚自 v" + versionNo, STATUS_PUBLISHED);
        cacheInvalidator.evictAgentCaches(agent.getAgentCode(), agentId);
        Map<String, Object> afterValue = new LinkedHashMap<>();
        afterValue.put("from_version_no", versionNo);
        afterValue.put("to_version_no", version.getVersionNo());
        AiTxSupport.afterCommit(() -> auditLogService.recordAudit(operatorId, "ai_agent", agentId, "rollback",
                "ai_agent", null, afterValue, null, null));
        return version.getVersionNo();
    }

    public IPage<AiAgentVersionVO> listVersions(Long agentId, int pageNum, int pageSize) {
        Page<SysAiAgentVersion> page = new Page<>(pageNum, pageSize);
        IPage<SysAiAgentVersion> versionPage = versionMapper.selectVersionPage(page, agentId);
        Page<AiAgentVersionVO> result = new Page<>(pageNum, pageSize, versionPage.getTotal());
        result.setRecords(new ArrayList<>(versionPage.getRecords().stream().map(this::toVO).toList()));
        return result;
    }

    /**
     * 版本差异对比：递归比较两个快照，仅记录叶节点差异（列表视为整体，子 Agent/Skills 顺序敏感）
     */
    public List<Map<String, Object>> diffVersions(Long agentId, Integer baseVersionNo, Integer targetVersionNo) {
        Map<String, Object> base = loadSnapshot(agentId, baseVersionNo);
        Map<String, Object> target = loadSnapshot(agentId, targetVersionNo);
        List<Map<String, Object>> diffs = new ArrayList<>();
        diff(base, target, "", diffs);
        return diffs;
    }

    /**
     * 版本详情：元数据 + 发布快照（config 替换为冻结的 resolved_config，保证运行面行为可复现）
     */
    public AiAgentVersionVO getVersionDetail(Long agentId, Integer versionNo) {
        SysAiAgentVersion version = versionMapper.getByAgentAndVersion(agentId, versionNo);
        if (version == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在");
        }
        AiAgentVersionVO vo = toVO(version);
        vo.setSnapshot(resolveSnapshot(version.getSnapshot() == null ? Map.of() : version.getSnapshot()));
        return vo;
    }

    /**
     * 已发布版本快照（未指定版本号取当前已发布版本）
     */
    public Map<String, Object> getPublishedSnapshot(Long agentId, Integer versionNo) {
        SysAiAgentVersion version = versionNo == null
                ? versionMapper.getLatestPublished(agentId)
                : versionMapper.getByAgentAndVersion(agentId, versionNo);
        if (version == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND,
                    versionNo == null ? "该 Agent 暂无已发布版本" : "版本快照不存在");
        }
        return resolveSnapshot(version.getSnapshot() == null ? Map.of() : version.getSnapshot());
    }

    /**
     * 快照生效配置：config 字段替换为发布时冻结的 resolved_config（不依赖运行时 sys_dict 再合并）
     */
    public Map<String, Object> resolveSnapshot(Map<String, Object> snapshot) {
        Map<String, Object> resolved = new LinkedHashMap<>(snapshot);
        if (snapshot.get("resolved_config") instanceof Map) {
            resolved.put("config", snapshot.get("resolved_config"));
        }
        return resolved;
    }

    /**
     * 写入版本记录：序列化可编辑态为快照 + 版本号 MAX+1，撞唯一键按冲突号递增重试
     */
    private SysAiAgentVersion writeVersion(SysAiAgent agent, Long operatorId, String changeNote, int status) {
        Map<String, Object> snapshot = buildSnapshot(agent);
        int versionNo = versionMapper.nextVersionNo(agent.getId());
        for (int attempt = 0; attempt < VERSION_NO_CONFLICT_RETRY; attempt++) {
            SysAiAgentVersion version = new SysAiAgentVersion();
            version.setAgentId(agent.getId());
            version.setVersionNo(versionNo);
            version.setSnapshot(snapshot);
            version.setStatus(status);
            version.setChangeNote(changeNote == null ? "" : changeNote);
            version.setOperatorId(operatorId);
            version.setCreateTime(LocalDateTime.now());
            try {
                versionMapper.insert(version);
                return version;
            } catch (DuplicateKeyException e) {
                log.warn("版本号 {} 冲突(agent={})，重试 {}", versionNo, agent.getId(), attempt + 1);
                versionNo++;
            }
        }
        throw new BusinessException(ResultCode.DATA_EXISTS, "版本号并发冲突，请重试发布");
    }

    /**
     * 序列化主表可编辑态为版本快照，冻结"继承默认"语义（resolved_config）
     */
    private Map<String, Object> buildSnapshot(SysAiAgent agent) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("name", agent.getName());
        snapshot.put("description", agent.getDescription());
        snapshot.put("system_prompt", agent.getSystemPrompt());
        snapshot.put("model_id", agent.getModelId());
        snapshot.put("reasoning_mode", agent.getReasoningMode());
        snapshot.put("config", agent.getConfig());
        snapshot.put("resolved_config", configResolver.resolve(agent.getConfig()));
        snapshot.put("permissions", agent.getPermissions());
        snapshot.put("is_subagent", agent.getIsSubagent());
        snapshot.put("is_team", agent.getIsTeam());
        snapshot.put("is_exposed", agent.getIsExposed());
        snapshot.put("skills", skillMapper.selectList(new LambdaQueryWrapper<SysAiAgentSkill>()
                        .eq(SysAiAgentSkill::getAgentId, agent.getId()))
                .stream().map(SysAiAgentSkill::getSkillName).toList());
        snapshot.put("mcp_namespaces", mcpMapper.selectList(new LambdaQueryWrapper<SysAiAgentMcp>()
                        .eq(SysAiAgentMcp::getAgentId, agent.getId()))
                .stream().map(SysAiAgentMcp::getMcpNamespace).toList());
        List<Map<String, Object>> subagents = new ArrayList<>();
        for (SysAiAgentSubagent link : subagentMapper.selectList(new LambdaQueryWrapper<SysAiAgentSubagent>()
                .eq(SysAiAgentSubagent::getParentAgentId, agent.getId()))) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("agent_id", link.getSubagentAgentId());
            item.put("priority", link.getPriority());
            item.put("endpoint_id", link.getEndpointId());
            subagents.add(item);
        }
        snapshot.put("subagents", subagents);
        return snapshot;
    }

    /**
     * 关联关系覆盖式恢复（回滚用）
     */
    private void replaceRelations(Long agentId, Map<String, Object> snapshot) {
        skillMapper.delete(new LambdaQueryWrapper<SysAiAgentSkill>().eq(SysAiAgentSkill::getAgentId, agentId));
        for (Object name : asList(snapshot.get("skills"))) {
            SysAiAgentSkill link = new SysAiAgentSkill();
            link.setAgentId(agentId);
            link.setSkillName(String.valueOf(name));
            link.setCreateTime(LocalDateTime.now());
            skillMapper.insert(link);
        }
        mcpMapper.delete(new LambdaQueryWrapper<SysAiAgentMcp>().eq(SysAiAgentMcp::getAgentId, agentId));
        for (Object namespace : asList(snapshot.get("mcp_namespaces"))) {
            SysAiAgentMcp link = new SysAiAgentMcp();
            link.setAgentId(agentId);
            link.setMcpNamespace(String.valueOf(namespace));
            link.setCreateTime(LocalDateTime.now());
            mcpMapper.insert(link);
        }
        subagentMapper.delete(new LambdaQueryWrapper<SysAiAgentSubagent>()
                .eq(SysAiAgentSubagent::getParentAgentId, agentId));
        for (Object item : asList(snapshot.get("subagents"))) {
            Map<String, Object> map = asMap(item);
            if (map == null || map.get("agent_id") == null) {
                continue;
            }
            SysAiAgentSubagent link = new SysAiAgentSubagent();
            link.setParentAgentId(agentId);
            link.setSubagentAgentId(((Number) map.get("agent_id")).longValue());
            link.setPriority(map.get("priority") == null ? 0 : ((Number) map.get("priority")).intValue());
            link.setEndpointId(map.get("endpoint_id") == null ? null : ((Number) map.get("endpoint_id")).longValue());
            link.setCreateTime(LocalDateTime.now());
            subagentMapper.insert(link);
        }
    }

    private Map<String, Object> loadSnapshot(Long agentId, Integer versionNo) {
        SysAiAgentVersion version = versionMapper.getByAgentAndVersion(agentId, versionNo);
        if (version == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本 " + versionNo + " 不存在");
        }
        return version.getSnapshot() == null ? Map.of() : version.getSnapshot();
    }

    @SuppressWarnings("unchecked")
    private void diff(Object baseVal, Object targetVal, String prefix, List<Map<String, Object>> acc) {
        if (baseVal instanceof Map && targetVal instanceof Map) {
            Set<String> keys = new TreeSet<>();
            keys.addAll(((Map<String, Object>) baseVal).keySet());
            keys.addAll(((Map<String, Object>) targetVal).keySet());
            for (String key : keys) {
                diff(((Map<String, Object>) baseVal).get(key), ((Map<String, Object>) targetVal).get(key),
                        prefix.isEmpty() ? key : prefix + "." + key, acc);
            }
            return;
        }
        if (baseVal instanceof List && targetVal instanceof List) {
            if (!baseVal.equals(targetVal)) {
                acc.add(diffItem(prefix, baseVal, targetVal));
            }
            return;
        }
        if (!Objects.equals(baseVal, targetVal)) {
            acc.add(diffItem(prefix, baseVal, targetVal));
        }
    }

    private Map<String, Object> diffItem(String field, Object base, Object target) {
        Map<String, Object> item = new TreeMap<>();
        item.put("field", field);
        item.put("base", base);
        item.put("target", target);
        return item;
    }

    private AiAgentVersionVO toVO(SysAiAgentVersion version) {
        AiAgentVersionVO vo = new AiAgentVersionVO();
        vo.setId(version.getId());
        vo.setAgentId(version.getAgentId());
        vo.setVersionNo(version.getVersionNo());
        vo.setStatus(version.getStatus());
        vo.setChangeNote(version.getChangeNote());
        vo.setOperatorId(version.getOperatorId());
        vo.setCreateTime(version.getCreateTime());
        return vo;
    }

    private String stringValue(Map<String, Object> snapshot, String key, String fallback) {
        Object value = snapshot.get(key);
        return value == null ? fallback : String.valueOf(value);
    }

    private Integer intValue(Map<String, Object> snapshot, String key, Integer fallback) {
        Object value = snapshot.get(key);
        return value instanceof Number number ? number.intValue() : fallback;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> asMap(Object value) {
        return value instanceof Map ? (Map<String, Object>) value : null;
    }

    @SuppressWarnings("unchecked")
    private List<Object> asList(Object value) {
        return value instanceof List ? (List<Object>) value : List.of();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> asMapList(Object value) {
        return value instanceof List ? (List<Map<String, Object>>) value : List.of();
    }
}
