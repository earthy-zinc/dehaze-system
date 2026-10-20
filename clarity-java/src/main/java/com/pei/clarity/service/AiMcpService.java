package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysAiMcpServer;
import com.pei.clarity.model.form.McpCredentialForm;
import com.pei.clarity.model.form.McpNamespaceForm;
import com.pei.clarity.model.form.McpServerForm;
import com.pei.clarity.model.form.McpServerUpdateForm;
import com.pei.clarity.model.query.McpCallPageQuery;
import com.pei.clarity.model.query.McpServerPageQuery;
import com.pei.clarity.model.vo.McpCallVO;
import com.pei.clarity.model.vo.McpMarketPresetVO;
import com.pei.clarity.model.vo.McpNamespaceVO;
import com.pei.clarity.model.vo.McpServerVO;
import java.util.List;

/** 外部 MCP Server 管理（A 类：注册表 CRUD/启停/命名空间/凭据/市场/调用审计） */
public interface AiMcpService extends IService<SysAiMcpServer> {

    Page<McpServerVO> listServers(McpServerPageQuery query);

    McpServerVO createServer(McpServerForm form);

    McpServerVO getServer(Long serverId);

    McpServerVO updateServer(Long serverId, McpServerUpdateForm form);

    void deleteServer(Long serverId);

    McpServerVO switchServerStatus(Long serverId, Integer status);

    List<McpNamespaceVO> listNamespaces(Long serverId);

    List<McpNamespaceVO> updateNamespaces(Long serverId, List<McpNamespaceForm> namespaces);

    void updateCredentials(Long serverId, McpCredentialForm form);

    List<McpMarketPresetVO> getMarket();

    McpServerVO installPreset(String presetId);

    Page<McpCallVO> listCalls(McpCallPageQuery query);
}
