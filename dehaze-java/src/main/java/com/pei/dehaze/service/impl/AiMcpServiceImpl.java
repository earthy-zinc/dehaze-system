package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.common.util.AiJsonUtils;
import com.pei.dehaze.mapper.SysAiMcpCallMapper;
import com.pei.dehaze.mapper.SysAiMcpNamespaceMapper;
import com.pei.dehaze.mapper.SysAiMcpServerMapper;
import com.pei.dehaze.mapper.SysAiMcpToolMapper;
import com.pei.dehaze.model.entity.SysAiMcpCall;
import com.pei.dehaze.model.entity.SysAiMcpNamespace;
import com.pei.dehaze.model.entity.SysAiMcpServer;
import com.pei.dehaze.model.entity.SysAiMcpTool;
import com.pei.dehaze.model.form.McpCredentialForm;
import com.pei.dehaze.model.form.McpNamespaceForm;
import com.pei.dehaze.model.form.McpServerForm;
import com.pei.dehaze.model.form.McpServerUpdateForm;
import com.pei.dehaze.model.query.McpCallPageQuery;
import com.pei.dehaze.model.query.McpServerPageQuery;
import com.pei.dehaze.model.vo.McpCallVO;
import com.pei.dehaze.model.vo.McpMarketPresetVO;
import com.pei.dehaze.model.vo.McpNamespaceVO;
import com.pei.dehaze.model.vo.McpServerVO;
import com.pei.dehaze.service.AiCacheInvalidator;
import com.pei.dehaze.service.AiMcpService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * 外部 MCP Server 管理实现。
 *
 * <p>工具拉取/试调用/健康探测属 B 类（由 proxy 承担），本服务只做注册表与配置管理。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiMcpServiceImpl extends ServiceImpl<SysAiMcpServerMapper, SysAiMcpServer> implements AiMcpService {

    /** 命名空间名约束：外部工具运行时命名为 <namespace>_<tool>，须是合法工具名片段 */
    private static final Pattern NAMESPACE_NAME = Pattern.compile("^[a-zA-Z][a-zA-Z0-9_-]{0,63}$");
    private static final int MAX_TOOL_NAME_LENGTH = 256;
    /** 拒绝命名空间时回显的可用工具数上限，避免工具清单上百条时错误信息过长 */
    private static final int HINT_TOOL_LIMIT = 20;

    /**
     * python {@code McpServerCreate/McpServerUpdate.protocol_type} 为 {@code Literal["streamable-http","sse"]}，
     * 非法字面量在 python 属请求校验阶段（A0400）。白名单取值校验保持 service 层单一信息源（不搬 @Pattern 到 DTO），
     * 否则拼错的协议名会直接落库并在后续探测/装载时静默失败。
     */
    private static final Set<String> PROTOCOL_TYPES = Set.of("streamable-http", "sse");

    /** MCP 市场预设目录（内置静态配置，与 python mcp_presets.MARKET_PRESETS 同源） */
    private static final List<Map<String, Object>> MARKET_PRESETS = List.of(
            Map.of("presetId", "github", "name", "GitHub",
                    "description", "GitHub 仓库/Issue/PR/代码管理",
                    "capabilityTags", List.of("github", "repo", "issue", "code"),
                    "protocolType", "streamable-http",
                    "endpoint", "https://api.githubcopilot.com/mcp/",
                    "authType", "oauth2"),
            Map.of("presetId", "mysql", "name", "MySQL",
                    "description", "MySQL 数据库查询与运维",
                    "capabilityTags", List.of("database", "mysql", "sql"),
                    "protocolType", "streamable-http",
                    "endpoint", "http://127.0.0.1:8083/mcp",
                    "authType", "api_key"),
            Map.of("presetId", "search", "name", "网络搜索",
                    "description", "联网搜索与网页摘要获取",
                    "capabilityTags", List.of("search", "web", "browser"),
                    "protocolType", "streamable-http",
                    "endpoint", "http://127.0.0.1:8084/mcp",
                    "authType", "api_key"));

    private final SysAiMcpNamespaceMapper namespaceMapper;
    private final SysAiMcpToolMapper toolMapper;
    private final SysAiMcpCallMapper callMapper;
    private final AiCredentialCipher cipher;
    private final AiCacheInvalidator cacheInvalidator;

    // ==================== Server 注册表 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<McpServerVO> listServers(McpServerPageQuery query) {
        String keyword = query.getKeyword();
        LambdaQueryWrapper<SysAiMcpServer> wrapper = new LambdaQueryWrapper<SysAiMcpServer>()
                .and(CharSequenceUtil.isNotBlank(keyword), q -> q
                        .like(SysAiMcpServer::getName, keyword)
                        .or()
                        .like(SysAiMcpServer::getDescription, keyword))
                .eq(query.getStatus() != null, SysAiMcpServer::getStatus, query.getStatus())
                .orderByDesc(SysAiMcpServer::getId);
        Page<SysAiMcpServer> page = this.page(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);
        Page<McpServerVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toVO).toList());
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public McpServerVO createServer(McpServerForm form) {
        if (form.getProtocolType() == null || !PROTOCOL_TYPES.contains(form.getProtocolType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "仅支持 streamable-http / sse 传输协议");
        }
        if (getActiveByName(form.getName()) != null) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在");
        }
        if (CharSequenceUtil.isBlank(form.getEndpoint())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "端点 URL 不能为空");
        }
        SysAiMcpServer server = new SysAiMcpServer();
        server.setName(form.getName());
        server.setDescription(form.getDescription());
        server.setProtocolType(form.getProtocolType());
        server.setEndpoint(form.getEndpoint());
        server.setAuthType(form.getAuthType());
        server.setStatus(1);
        server.setToolCount(0);
        this.save(server);
        return toVO(server);
    }

    @Override
    @Transactional(readOnly = true)
    public McpServerVO getServer(Long serverId) {
        return toVO(getOrRaise(serverId));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public McpServerVO updateServer(Long serverId, McpServerUpdateForm form) {
        if (form.getProtocolType() != null && !PROTOCOL_TYPES.contains(form.getProtocolType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "仅支持 streamable-http / sse 传输协议");
        }
        SysAiMcpServer server = getOrRaise(serverId);
        if (CharSequenceUtil.isNotBlank(form.getName()) && !form.getName().equals(server.getName())) {
            SysAiMcpServer existing = getActiveByName(form.getName());
            if (existing != null && !existing.getId().equals(serverId)) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "MCP Server 名称已存在");
            }
            server.setName(form.getName());
        }
        if (form.getEndpoint() != null) {
            if (CharSequenceUtil.isBlank(form.getEndpoint())) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "端点 URL 不能为空");
            }
            server.setEndpoint(form.getEndpoint());
        }
        if (form.getDescription() != null) {
            server.setDescription(form.getDescription());
        }
        if (form.getProtocolType() != null) {
            server.setProtocolType(form.getProtocolType());
        }
        if (form.getAuthType() != null) {
            server.setAuthType(form.getAuthType());
        }
        this.updateById(server);
        cacheInvalidator.evictReasoningGraphs();
        return toVO(server);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteServer(Long serverId) {
        SysAiMcpServer server = getOrRaise(serverId);
        long refs = this.baseMapper.countAgentReferences(server.getId());
        if (refs > 0) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                    "MCP Server [" + server.getName() + "] 已被 " + refs
                            + " 个 Agent 关联（命名空间），请先解绑再删除");
        }
        this.removeById(server.getId());
        cacheInvalidator.evictReasoningGraphs();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public McpServerVO switchServerStatus(Long serverId, Integer status) {
        SysAiMcpServer server = getOrRaise(serverId);
        server.setStatus(status);
        this.updateById(server);
        cacheInvalidator.evictReasoningGraphs();
        return toVO(server);
    }

    // ==================== 命名空间 ====================

    @Override
    @Transactional(readOnly = true)
    public List<McpNamespaceVO> listNamespaces(Long serverId) {
        getOrRaise(serverId);
        return namespaceMapper.selectList(new LambdaQueryWrapper<SysAiMcpNamespace>()
                        .eq(SysAiMcpNamespace::getServerId, serverId)
                        .orderByAsc(SysAiMcpNamespace::getId))
                .stream()
                .map(this::toNamespaceVO)
                .toList();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<McpNamespaceVO> updateNamespaces(Long serverId, List<McpNamespaceForm> namespaces) {
        getOrRaise(serverId);
        Set<String> known = knownToolNames(serverId);
        Set<String> seen = new LinkedHashSet<>();
        for (McpNamespaceForm item : namespaces) {
            if (!NAMESPACE_NAME.matcher(item.getName() == null ? "" : item.getName()).matches()) {
                throw new BusinessException(ResultCode.PARAM_ERROR,
                        "命名空间名非法（须字母开头，仅含字母/数字/下划线/连字符，≤64字符）: " + item.getName());
            }
            if (!seen.add(item.getName())) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "命名空间重复: " + item.getName());
            }
            for (String tool : item.getToolNames()) {
                if (CharSequenceUtil.isBlank(tool) || tool.length() > MAX_TOOL_NAME_LENGTH) {
                    throw new BusinessException(ResultCode.PARAM_ERROR,
                            "命名空间 " + item.getName() + " 的工具名不能为空且≤256字符");
                }
            }
            // 拒绝清单外的工具名：拼错的工具名装载期静默失效，且无提示可自查
            List<String> unknown = item.getToolNames().stream()
                    .filter(tool -> !known.contains(tool))
                    .toList();
            if (!unknown.isEmpty()) {
                String hint = known.isEmpty()
                        ? "无（请先在工具 Tab 拉取清单）"
                        : String.join("、", known.stream().sorted().limit(HINT_TOOL_LIMIT).toList());
                throw new BusinessException(ResultCode.PARAM_ERROR,
                        "命名空间 " + item.getName() + " 含未拉取到的工具: " + String.join(", ", unknown)
                                + "；当前可用工具: " + hint);
            }
        }
        namespaceMapper.delete(new LambdaQueryWrapper<SysAiMcpNamespace>()
                .eq(SysAiMcpNamespace::getServerId, serverId));
        for (McpNamespaceForm item : namespaces) {
            SysAiMcpNamespace entity = new SysAiMcpNamespace();
            entity.setServerId(serverId);
            entity.setNamespace(item.getName());
            entity.setToolNames(AiJsonUtils.write(item.getToolNames()));
            namespaceMapper.insert(entity);
        }
        cacheInvalidator.evictReasoningGraphs();
        return listNamespaces(serverId);
    }

    // ==================== 凭据 ====================

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void updateCredentials(Long serverId, McpCredentialForm form) {
        SysAiMcpServer server = getOrRaise(serverId);
        if (Boolean.TRUE.equals(form.getClear())) {
            server.setCredentials(null);
        } else {
            Map<String, Object> credentials = readCredentials(server.getCredentials());
            if (CharSequenceUtil.isNotBlank(form.getApiKey())) {
                credentials.put("api_key", cipher.encrypt(form.getApiKey()));
            }
            if (form.getExtra() != null && !form.getExtra().isEmpty()) {
                @SuppressWarnings("unchecked")
                Map<String, Object> extra = (Map<String, Object>) credentials.computeIfAbsent(
                        "extra", key -> new LinkedHashMap<String, Object>());
                form.getExtra().forEach((key, value) -> extra.put(key, cipher.encrypt(value)));
            }
            server.setCredentials(AiJsonUtils.write(credentials));
        }
        this.updateById(server);
        cacheInvalidator.evictReasoningGraphs();
    }

    // ==================== 市场 / 调用审计 ====================

    @Override
    @Transactional(readOnly = true)
    public List<McpMarketPresetVO> getMarket() {
        List<McpMarketPresetVO> items = new ArrayList<>(MARKET_PRESETS.size());
        for (Map<String, Object> preset : MARKET_PRESETS) {
            McpMarketPresetVO vo = new McpMarketPresetVO();
            vo.setPresetId((String) preset.get("presetId"));
            vo.setName((String) preset.get("name"));
            vo.setDescription((String) preset.get("description"));
            @SuppressWarnings("unchecked")
            List<String> tags = (List<String>) preset.get("capabilityTags");
            vo.setCapabilityTags(tags);
            vo.setInstalled(getActiveByName((String) preset.get("name")) != null);
            items.add(vo);
        }
        return items;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public McpServerVO installPreset(String presetId) {
        Map<String, Object> preset = MARKET_PRESETS.stream()
                .filter(item -> presetId.equals(item.get("presetId")))
                .findFirst()
                .orElseThrow(() -> new BusinessException(ResultCode.PARAM_ERROR, "未知的市场预设"));
        SysAiMcpServer server = this.baseMapper.selectByNameIncludingDeleted((String) preset.get("name"));
        if (server == null) {
            server = new SysAiMcpServer();
            server.setName((String) preset.get("name"));
            server.setDescription((String) preset.get("description"));
            server.setProtocolType((String) preset.get("protocolType"));
            server.setEndpoint((String) preset.get("endpoint"));
            server.setAuthType((String) preset.get("authType"));
            server.setStatus(1);
            server.setToolCount(0);
            this.save(server);
        } else if (server.getDeleted() != null && server.getDeleted() != 0) {
            // 同名软删 Server 复活复用（唯一键 name 含软删行，无法新建同名）
            this.baseMapper.resurrect(server.getId());
            server.setDeleted(0L);
            server.setStatus(1);
        }
        // 工具清单拉取属 B 类（proxy），此处保持 Server 启用，由工具接口拉取后回填 tool_count
        return toVO(server);
    }

    @Override
    @Transactional(readOnly = true)
    public Page<McpCallVO> listCalls(McpCallPageQuery query) {
        LambdaQueryWrapper<SysAiMcpCall> wrapper = new LambdaQueryWrapper<SysAiMcpCall>()
                .eq(query.getServerId() != null, SysAiMcpCall::getServerId, query.getServerId())
                .eq(CharSequenceUtil.isNotBlank(query.getToolName()), SysAiMcpCall::getToolName, query.getToolName())
                .orderByDesc(SysAiMcpCall::getCreateTime)
                .orderByDesc(SysAiMcpCall::getId);
        Page<SysAiMcpCall> page = callMapper.selectPage(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);
        Page<McpCallVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        List<McpCallVO> records = new ArrayList<>(page.getRecords().size());
        for (SysAiMcpCall call : page.getRecords()) {
            McpCallVO vo = new McpCallVO();
            vo.setId(call.getId());
            vo.setUserId(call.getUserId());
            vo.setServerId(call.getServerId());
            vo.setServerName(call.getServerName());
            vo.setToolName(call.getToolName());
            vo.setResult(call.getResult());
            vo.setLatencyMs(call.getLatencyMs());
            vo.setCreateTime(call.getCreateTime());
            records.add(vo);
        }
        result.setRecords(records);
        return result;
    }

    // ==================== 内部实现 ====================

    private SysAiMcpServer getOrRaise(Long serverId) {
        SysAiMcpServer server = this.getById(serverId);
        if (server == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "MCP Server 不存在");
        }
        return server;
    }

    private SysAiMcpServer getActiveByName(String name) {
        return this.getOne(new LambdaQueryWrapper<SysAiMcpServer>()
                .eq(SysAiMcpServer::getName, name)
                .orderByAsc(SysAiMcpServer::getId), false);
    }

    private Set<String> knownToolNames(Long serverId) {
        return toolMapper.selectList(new LambdaQueryWrapper<SysAiMcpTool>()
                        .eq(SysAiMcpTool::getServerId, serverId)
                        .select(SysAiMcpTool::getName))
                .stream()
                .map(SysAiMcpTool::getName)
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    }

    private Map<String, Object> readCredentials(String json) {
        Map<String, Object> credentials = AiJsonUtils.read(json, new TypeReference<Map<String, Object>>() {
        });
        return credentials == null ? new LinkedHashMap<>() : new LinkedHashMap<>(credentials);
    }

    private McpNamespaceVO toNamespaceVO(SysAiMcpNamespace entity) {
        McpNamespaceVO vo = new McpNamespaceVO();
        vo.setName(entity.getNamespace());
        List<String> toolNames = AiJsonUtils.read(entity.getToolNames(), new TypeReference<List<String>>() {
        });
        vo.setToolNames(toolNames == null ? List.of() : toolNames);
        return vo;
    }

    private McpServerVO toVO(SysAiMcpServer server) {
        McpServerVO vo = new McpServerVO();
        vo.setId(server.getId());
        vo.setName(server.getName());
        vo.setDescription(server.getDescription());
        vo.setProtocolType(server.getProtocolType());
        vo.setEndpoint(server.getEndpoint());
        vo.setAuthType(server.getAuthType());
        vo.setStatus(server.getStatus());
        vo.setHealth(server.getHealth());
        vo.setLastCheckTime(server.getLastCheckTime());
        vo.setToolCount(server.getToolCount());
        vo.setCredentialConfigured(CharSequenceUtil.isNotBlank(server.getCredentials()));
        vo.setCreateTime(server.getCreateTime());
        vo.setUpdateTime(server.getUpdateTime());
        return vo;
    }
}
