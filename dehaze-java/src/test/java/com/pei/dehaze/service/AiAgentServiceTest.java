package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.apache.ibatis.builder.MapperBuilderAssistant;
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
import com.pei.dehaze.model.entity.SysAiAgentSubagent;
import com.pei.dehaze.model.form.AiAgentCreateForm;
import com.pei.dehaze.model.form.AiAgentSubAgentsForm;
import com.pei.dehaze.model.form.AiAgentUpdateForm;
import com.pei.dehaze.model.form.AiSubAgentItemForm;
import com.pei.dehaze.model.query.AiAgentPageQuery;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.function.Executable;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 智能体管理单测：默认 Agent 保护、删除引用校验与级联清理、关联覆盖式校验、子 Agent 环检测。
 *
 * <p>这些守卫是"删不掉的 Agent / 无限递归的子代理"两类线上事故的唯一边界，需逐条钉住。
 */
@DisplayName("AiAgentService 守卫与级联")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiAgentServiceTest {

    @Mock
    private SysAiAgentMapper agentMapper;
    @Mock
    private SysAiAgentVersionMapper agentVersionMapper;
    @Mock
    private SysAiAgentSkillMapper agentSkillMapper;
    @Mock
    private SysAiAgentMcpMapper agentMcpMapper;
    @Mock
    private SysAiAgentSubagentMapper agentSubagentMapper;
    @Mock
    private SysAiAgentEvalDatasetMapper evalDatasetMapper;
    @Mock
    private SysAiAgentEvalSampleMapper evalSampleMapper;
    @Mock
    private SysAiAgentEvalRunMapper evalRunMapper;
    @Mock
    private SysAiConversationMapper conversationMapper;
    @Mock
    private AiInsightMapper insightMapper;
    @Mock
    private AiCacheInvalidator cacheInvalidator;
    @Mock
    private AuditLogService auditLogService;

    @InjectMocks
    private AiAgentService service;

    @BeforeEach
    void setUp() {
        when(agentSkillMapper.selectList(any(Wrapper.class))).thenReturn(List.of());
        when(agentMcpMapper.selectList(any(Wrapper.class))).thenReturn(List.of());
        when(agentSubagentMapper.selectList(any(Wrapper.class))).thenReturn(List.of());
    }

    /**
     * LambdaQueryWrapper 的列名解析依赖实体 TableInfo（正常由 MyBatis 启动期注册），
     * 纯 Mockito 单测需手动初始化，否则 {@code .select(SysAiAgent::getId)} 抛 lambda cache 缺失。
     */
    @BeforeAll
    static void initTableInfo() {
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), ""), SysAiAgent.class);
    }

    private SysAiAgent agent(Long id, String code) {
        SysAiAgent agent = new SysAiAgent();
        agent.setId(id);
        agent.setAgentCode(code);
        agent.setName("Agent-" + code);
        agent.setStatus(1);
        agent.setIsSubagent(0);
        agent.setIsTeam(0);
        agent.setIsExposed(0);
        return agent;
    }

    private AiAgentSubAgentsForm subagents(Long... childIds) {
        AiAgentSubAgentsForm form = new AiAgentSubAgentsForm();
        form.setSubagents(Arrays.stream(childIds).map(id -> {
            AiSubAgentItemForm item = new AiSubAgentItemForm();
            item.setAgentId(id);
            item.setPriority(1);
            return item;
        }).toList());
        return form;
    }

    private void assertBizError(ResultCode expected, Executable action) {
        assertThat(assertThrows(BusinessException.class, action).getResultCode()).isEqualTo(expected);
    }

    @Test
    @DisplayName("创建 Agent：编码重复报 A0501")
    void createRejectsDuplicateCode() {
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(agent(1L, "dup"));

        AiAgentCreateForm form = new AiAgentCreateForm();
        form.setAgentCode("dup");
        form.setName("重名");
        form.setModelId("qwen3-0.6b");

        assertBizError(ResultCode.DATA_EXISTS, () -> service.create(form));
        verify(agentMapper, never()).insert(any(SysAiAgent.class));
    }

    @Test
    @DisplayName("创建 Agent：默认值落库（status=1）并失效可选列表缓存")
    void createFillsDefaultsAndEvictsEnabledList() {
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        AiAgentCreateForm form = new AiAgentCreateForm();
        form.setAgentCode("newbie");
        form.setName("新 Agent");
        form.setModelId("qwen3-0.6b");

        assertThat(service.create(form).getStatus()).isEqualTo(1);
        verify(agentMapper).insert(any(SysAiAgent.class));
        verify(cacheInvalidator).evict("ai:agent:list:enabled");
    }

    @Test
    @DisplayName("删除默认 Agent 报 A0503（系统预置不可删）")
    void deleteDefaultAgentRejected() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "default"));

        assertBizError(ResultCode.OPERATION_NOT_ALLOW, () -> service.delete(1L));
        verify(agentMapper, never()).deleteById(any(Long.class));
    }

    @Test
    @DisplayName("删除被会话使用的 Agent 报 A0504 并回报占用会话数")
    void deleteAgentWithConversationsRejected() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "used"));
        when(conversationMapper.countByAgentCode("used")).thenReturn(3L);

        assertThatThrownBy(() -> service.delete(1L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("3 个会话");
        verify(agentMapper, never()).deleteById(any(Long.class));
    }

    @Test
    @DisplayName("删除被引用为子 Agent 的 Agent 报 A0504")
    void deleteAgentReferencedAsSubagentRejected() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "child"));
        when(conversationMapper.countByAgentCode("child")).thenReturn(0L);
        when(insightMapper.countSubagentReferences(1L)).thenReturn(2L);

        assertBizError(ResultCode.DATA_BIND_EXISTS, () -> service.delete(1L));
        verify(agentMapper, never()).deleteById(any(Long.class));
    }

    @Test
    @DisplayName("删除 Agent：级联清理评测资产（集软删/样本物理删/执行记录物理删）并留审计")
    void deleteCascadesEvalAssets() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "dead"));
        when(conversationMapper.countByAgentCode("dead")).thenReturn(0L);
        when(insightMapper.countSubagentReferences(1L)).thenReturn(0L);
        SysAiAgentEvalDataset dataset = new SysAiAgentEvalDataset();
        dataset.setId(11L);
        when(evalDatasetMapper.selectList(any(Wrapper.class))).thenReturn(List.of(dataset));
        when(evalSampleMapper.deleteByDatasetIds(List.of(11L))).thenReturn(5);
        when(evalRunMapper.deleteByAgentId(1L)).thenReturn(2);

        service.delete(1L);

        verify(evalSampleMapper).deleteByDatasetIds(List.of(11L));
        verify(evalDatasetMapper).softDeleteByIds(List.of(11L));
        verify(evalRunMapper).deleteByAgentId(1L);
        verify(agentMapper).deleteById(1L);
        verify(cacheInvalidator).evictAgentCaches("dead", 1L);
        verify(auditLogService).recordAudit(isNull(), eq("ai_agent"), eq(1L), eq("delete"), eq("ai_agent"),
                any(), any(), isNull(), isNull());
    }

    @Test
    @DisplayName("复制 Agent：目标编码已存在报 A0501")
    void copyRejectsExistingTargetCode() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "src"));
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(agent(2L, "target"));

        assertBizError(ResultCode.DATA_EXISTS, () -> service.copy(1L, "target"));
        verify(agentMapper, never()).insert(any(SysAiAgent.class));
    }

    @Test
    @DisplayName("设置 Skills：缺失名称一次性报 A0401 且不破坏既有绑定")
    void setSkillsRejectsMissingNames() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        when(agentSkillMapper.listExistingSkillNames(List.of("s1", "s2"))).thenReturn(List.of("s1"));

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.setSkills(1L, List.of("s1", "s2")));
        verify(agentSkillMapper, never()).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("设置 MCP 命名空间：未注册命名空间报 A0401")
    void setMcpNamespacesRejectsUnregisteredNamespace() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        when(agentMcpMapper.listRegisteredNamespaces(List.of("nope"))).thenReturn(List.of());

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.setMcpNamespaces(1L, List.of("nope")));
        verify(agentMcpMapper, never()).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("设置子 Agent：自引用报 A0400")
    void setSubagentsRejectsSelfReference() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));

        assertBizError(ResultCode.PARAM_ERROR, () -> service.setSubagents(1L, subagents(1L)));
        verify(agentSubagentMapper, never()).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("设置子 Agent：不存在的子 Agent 报 A0401")
    void setSubagentsRejectsMissingChild() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of());

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.setSubagents(1L, subagents(9L)));
        verify(agentSubagentMapper, never()).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("设置子 Agent：形成环（1→2→1）报 A0400，避免推理期子代理无限递归")
    void setSubagentsRejectsCycle() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(agent(2L, "a2")));
        SysAiAgentSubagent backEdge = new SysAiAgentSubagent();
        backEdge.setParentAgentId(2L);
        backEdge.setSubagentAgentId(1L);
        when(agentSubagentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(backEdge));

        assertThatThrownBy(() -> service.setSubagents(1L, subagents(2L)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("环");
        verify(agentSubagentMapper, never()).delete(any(Wrapper.class));
    }

    @Test
    @DisplayName("设置子 Agent：正常覆盖式写入并失效缓存")
    void setSubagentsReplacesBindings() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(agent(2L, "a2")));

        service.setSubagents(1L, subagents(2L));

        verify(agentSubagentMapper).delete(any(Wrapper.class));
        verify(agentSubagentMapper).insert(any(SysAiAgentSubagent.class));
        verify(cacheInvalidator).evictAgentCaches("a1", 1L);
    }

    @Test
    @DisplayName("列表：关联计数按批量查询回填（不做逐 Agent N+1）")
    void listFillsRelationCountsInBatch() {
        Page<SysAiAgent> page = new Page<>(1, 10, 1);
        page.setRecords(List.of(agent(1L, "a1")));
        when(agentMapper.selectPage(any(), any(Wrapper.class))).thenReturn(page);
        when(insightMapper.countSkillsByAgentIds(anyList())).thenReturn(List.of());
        when(insightMapper.countMcpByAgentIds(anyList())).thenReturn(List.of());
        when(insightMapper.countSubagentsByAgentIds(anyList())).thenReturn(List.of());

        assertThat(service.list(new AiAgentPageQuery()).getRecords()).hasSize(1);
        verify(insightMapper).countSkillsByAgentIds(List.of(1L));
        verify(insightMapper).countMcpByAgentIds(List.of(1L));
        verify(insightMapper).countSubagentsByAgentIds(List.of(1L));
    }

    @Test
    @DisplayName("列表：分页为空时不发起关联计数 SQL")
    void listSkipsBatchQueriesWhenPageEmpty() {
        when(agentMapper.selectPage(any(), any(Wrapper.class))).thenReturn(new Page<>(1, 10, 0));

        assertThat(service.list(new AiAgentPageQuery()).getRecords()).isEmpty();
        verify(insightMapper, never()).countSkillsByAgentIds(anyList());
    }

    @Test
    @DisplayName("详情：子 Agent 关联带出名称与编码")
    void detailResolvesSubagentMetadata() {
        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        SysAiAgentSubagent link = new SysAiAgentSubagent();
        link.setParentAgentId(1L);
        link.setSubagentAgentId(2L);
        link.setPriority(3);
        when(agentSubagentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(link));
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(agent(2L, "a2")));

        var detail = service.getDetail(1L);

        assertThat(detail.getSubagents()).hasSize(1);
        assertThat(detail.getSubagents().get(0).getAgentCode()).isEqualTo("a2");
        assertThat(detail.getSubAgentCount()).isEqualTo(1);
    }

    @Test
    @DisplayName("推理范式白名单：非法值在 create/update 一律 A0400 且不落库")
    void invalidReasoningModeRejected() {
        // python 侧为 pattern ^(auto|direct|react|plan_execute|reflexion)$，非连续多值枚举 → service 层白名单
        AiAgentCreateForm create = new AiAgentCreateForm();
        create.setAgentCode("a1");
        create.setName("n");
        create.setModelId("m1");
        create.setReasoningMode("bogus");
        assertBizError(ResultCode.PARAM_ERROR, () -> service.create(create));

        when(agentMapper.selectById(1L)).thenReturn(agent(1L, "a1"));
        AiAgentUpdateForm update = new AiAgentUpdateForm();
        update.setReasoningMode("plan_execute_x");
        assertBizError(ResultCode.PARAM_ERROR, () -> service.update(1L, update));

        verify(agentMapper, never()).insert(any());
        verify(agentMapper, never()).updateById(any());
    }
}
