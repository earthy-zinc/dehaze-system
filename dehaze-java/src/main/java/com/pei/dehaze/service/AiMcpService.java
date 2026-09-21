package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysAiMcpServer;
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
