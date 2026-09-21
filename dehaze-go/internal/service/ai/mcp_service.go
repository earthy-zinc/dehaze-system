package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

// namespaceNamePattern 命名空间名约束：外部工具运行时命名为 <namespace>_<tool>，须是合法工具名片段
var namespaceNamePattern = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_-]{0,63}$`)

// hintToolLimit 拒绝命名空间时回显的可用工具数上限（工具清单可能上百条）
const hintToolLimit = 20

// mcpMarketPreset MCP 市场预设（内置静态配置，installed 由同名 Server 推导）
type mcpMarketPreset struct {
	PresetID       string   `json:"presetId"`
	Name           string   `json:"name"`
	Description    string   `json:"description"`
	CapabilityTags []string `json:"capabilityTags"`
	Installed      bool     `json:"installed"`
}

// marketPresets 与 dehaze-python app/service/ai_mcp/mcp_presets.py 同源
var marketPresets = []struct {
	PresetID       string
	Name           string
	Description    string
	CapabilityTags []string
	ProtocolType   string
	Endpoint       string
	AuthType       string
}{
	{"github", "GitHub", "GitHub 仓库/Issue/PR/代码管理", []string{"github", "repo", "issue", "code"}, "streamable-http", "https://api.githubcopilot.com/mcp/", "oauth2"},
	{"mysql", "MySQL", "MySQL 数据库查询与运维", []string{"database", "mysql", "sql"}, "streamable-http", "http://127.0.0.1:8083/mcp", "api_key"},
	{"search", "网络搜索", "联网搜索与网页摘要获取", []string{"search", "web", "browser"}, "streamable-http", "http://127.0.0.1:8084/mcp", "api_key"},
}

// McpServerService 外部 MCP Server 注册表/命名空间/凭据管理
type McpServerService struct {
	db   *gorm.DB
	repo *airepo.McpRepository
}

func NewMcpServerService(db *gorm.DB, repo *airepo.McpRepository) *McpServerService {
	return &McpServerService{db: db, repo: repo}
}

func (s *McpServerService) ListServers(ctx context.Context, q *bo.McpServerQuery) (*vo.PageResult[vo.McpServerVO], error) {
	page, size := q.PageNum, q.PageSize
	var status *int
	if q.Status != nil {
		value := int(*q.Status)
		status = &value
	}
	servers, total, err := s.repo.PaginateServers(ctx, page, size, q.Keyword, status)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 MCP Server 列表失败", err)
	}
	items := make([]vo.McpServerVO, 0, len(servers))
	for i := range servers {
		items = append(items, toMcpServerVO(&servers[i]))
	}
	return &vo.PageResult[vo.McpServerVO]{List: items, Total: total}, nil
}

func (s *McpServerService) CreateServer(ctx context.Context, form *bo.McpServerCreateForm, operatorID int64) (*vo.McpServerVO, error) {
	existing, err := s.repo.GetServerByName(ctx, form.Name, false)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "校验 Server 名称失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "MCP Server 名称已存在")
	}
	protocol := form.ProtocolType
	if protocol == "" {
		protocol = "streamable-http"
	}
	if protocol != "streamable-http" && protocol != "sse" {
		return nil, common.NewBizError(common.PARAM_ERROR, "仅支持 streamable-http / sse 传输协议")
	}
	if form.Endpoint == nil || strings.TrimSpace(*form.Endpoint) == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "端点 URL 不能为空")
	}
	server := &model.SysAiMcpServer{
		Name:         form.Name,
		Description:  form.Description,
		ProtocolType: protocol,
		Endpoint:     form.Endpoint,
		AuthType:     form.AuthType,
		Status:       1,
	}
	server.CreateBy = operatorID
	server.UpdateBy = operatorID
	if err := s.repo.CreateServer(ctx, server); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建 MCP Server 失败", err)
	}
	result := toMcpServerVO(server)
	return &result, nil
}

func (s *McpServerService) GetServer(ctx context.Context, serverID int64) (*vo.McpServerVO, error) {
	server, err := s.requireServer(ctx, serverID)
	if err != nil {
		return nil, err
	}
	result := toMcpServerVO(server)
	return &result, nil
}

func (s *McpServerService) UpdateServer(ctx context.Context, serverID int64, form *bo.McpServerUpdateForm, operatorID int64) (*vo.McpServerVO, error) {
	server, err := s.requireServer(ctx, serverID)
	if err != nil {
		return nil, err
	}
	updates := map[string]interface{}{"update_by": operatorID}
	if form.Name != nil {
		existing, findErr := s.repo.GetServerByName(ctx, *form.Name, false)
		if findErr != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "校验 Server 名称失败", findErr)
		}
		if existing != nil && existing.ID != server.ID {
			return nil, common.NewBizError(common.DATA_EXISTS, "MCP Server 名称已存在")
		}
		updates["name"] = *form.Name
	}
	if form.Description != nil {
		updates["description"] = *form.Description
	}
	if form.ProtocolType != nil {
		if *form.ProtocolType != "streamable-http" && *form.ProtocolType != "sse" {
			return nil, common.NewBizError(common.PARAM_ERROR, "仅支持 streamable-http / sse 传输协议")
		}
		updates["protocol_type"] = *form.ProtocolType
	}
	if form.Endpoint != nil {
		if strings.TrimSpace(*form.Endpoint) == "" {
			return nil, common.NewBizError(common.PARAM_ERROR, "端点 URL 不能为空")
		}
		updates["endpoint"] = *form.Endpoint
	}
	if form.AuthType != nil {
		updates["auth_type"] = *form.AuthType
	}
	if err := s.repo.UpdateServer(ctx, serverID, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新 MCP Server 失败", err)
	}
	updated, err := s.requireServer(ctx, serverID)
	if err != nil {
		return nil, err
	}
	invalidateReasoningGraphs(ctx)
	result := toMcpServerVO(updated)
	return &result, nil
}

func (s *McpServerService) DeleteServer(ctx context.Context, serverID, operatorID int64) error {
	server, err := s.requireServer(ctx, serverID)
	if err != nil {
		return err
	}
	refs, err := s.countAgentReferences(ctx, server.ID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "统计 Agent 关联失败", err)
	}
	if refs > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS,
			fmt.Sprintf("MCP Server [%s] 已被 %d 个 Agent 关联（命名空间），请先解绑再删除", server.Name, refs))
	}
	if err := s.repo.SoftDeleteServer(ctx, server.ID, operatorID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除 MCP Server 失败", err)
	}
	invalidateReasoningGraphs(ctx)
	return nil
}

func (s *McpServerService) SwitchServerStatus(ctx context.Context, serverID int64, status int8, operatorID int64) (*vo.McpServerVO, error) {
	if _, err := s.requireServer(ctx, serverID); err != nil {
		return nil, err
	}
	if err := s.repo.UpdateServer(ctx, serverID, map[string]interface{}{"status": status, "update_by": operatorID}); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新 MCP Server 状态失败", err)
	}
	updated, err := s.requireServer(ctx, serverID)
	if err != nil {
		return nil, err
	}
	invalidateReasoningGraphs(ctx)
	result := toMcpServerVO(updated)
	return &result, nil
}

// UpdateCredentials 合并式更新凭据：未传字段保留原值，clear=true 整体清空；仅存 AES 密文
func (s *McpServerService) UpdateCredentials(ctx context.Context, serverID int64, form *bo.McpCredentialForm, operatorID int64) error {
	server, err := s.requireServer(ctx, serverID)
	if err != nil {
		return err
	}
	var credentials map[string]any
	if form.Clear {
		credentials = nil
	} else {
		if len(server.Credentials) > 0 {
			if unmarshalErr := json.Unmarshal(server.Credentials, &credentials); unmarshalErr != nil {
				credentials = map[string]any{}
			}
		}
		if credentials == nil {
			credentials = map[string]any{}
		}
		if form.ApiKey != "" {
			cipherText, encErr := EncryptSecret(form.ApiKey)
			if encErr != nil {
				return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "凭据加密失败", encErr)
			}
			credentials["api_key"] = cipherText
		}
		if len(form.Extra) > 0 {
			extra, ok := credentials["extra"].(map[string]any)
			if !ok {
				extra = map[string]any{}
			}
			for key, value := range form.Extra {
				cipherText, encErr := EncryptSecret(value)
				if encErr != nil {
					return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "凭据加密失败", encErr)
				}
				extra[key] = cipherText
			}
			credentials["extra"] = extra
		}
	}
	payload, err := json.Marshal(credentials)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "凭据序列化失败", err)
	}
	updates := map[string]interface{}{"update_by": operatorID}
	if credentials == nil {
		updates["credentials"] = nil
	} else {
		updates["credentials"] = payload
	}
	if err := s.repo.UpdateServer(ctx, serverID, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新 MCP 凭据失败", err)
	}
	invalidateReasoningGraphs(ctx)
	return nil
}

// ==================== 命名空间 ====================

func (s *McpServerService) ListNamespaces(ctx context.Context, serverID int64) ([]vo.McpNamespaceVO, error) {
	if _, err := s.requireServer(ctx, serverID); err != nil {
		return nil, err
	}
	return s.listNamespaces(ctx, serverID)
}

func (s *McpServerService) UpdateNamespaces(ctx context.Context, serverID int64, forms []bo.McpNamespaceForm, operatorID int64) ([]vo.McpNamespaceVO, error) {
	if _, err := s.requireServer(ctx, serverID); err != nil {
		return nil, err
	}
	known, err := s.knownToolNames(ctx, serverID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询工具清单失败", err)
	}
	seen := make(map[string]bool, len(forms))
	items := make([]model.SysAiMcpNamespace, 0, len(forms))
	for _, form := range forms {
		if !namespaceNamePattern.MatchString(form.Name) {
			return nil, common.NewBizError(common.PARAM_ERROR,
				fmt.Sprintf("命名空间名非法（须字母开头，仅含字母/数字/下划线/连字符，≤64字符）: %s", form.Name))
		}
		if seen[form.Name] {
			return nil, common.NewBizError(common.PARAM_ERROR, "命名空间重复: "+form.Name)
		}
		seen[form.Name] = true
		for _, tool := range form.ToolNames {
			if tool == "" || len(tool) > 256 {
				return nil, common.NewBizError(common.PARAM_ERROR,
					fmt.Sprintf("命名空间 %s 的工具名不能为空且≤256字符", form.Name))
			}
		}
		unknown := make([]string, 0)
		for _, tool := range form.ToolNames {
			if !known[tool] {
				unknown = append(unknown, tool)
			}
		}
		if len(unknown) > 0 {
			available := make([]string, 0, len(known))
			for name := range known {
				available = append(available, name)
			}
			sort.Strings(available)
			if len(available) > hintToolLimit {
				available = available[:hintToolLimit]
			}
			hint := strings.Join(available, "、")
			if hint == "" {
				hint = "无（请先在工具 Tab 拉取清单）"
			}
			return nil, common.NewBizError(common.PARAM_ERROR,
				fmt.Sprintf("命名空间 %s 含未拉取到的工具: %s；当前可用工具: %s", form.Name, strings.Join(unknown, ", "), hint))
		}
		toolNames, marshalErr := json.Marshal(nonNilStrings(form.ToolNames))
		if marshalErr != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "工具清单序列化失败", marshalErr)
		}
		item := model.SysAiMcpNamespace{ServerID: serverID, Namespace: form.Name, ToolNames: toolNames}
		item.CreateBy = operatorID
		item.UpdateBy = operatorID
		items = append(items, item)
	}
	if err := s.repo.ReplaceNamespaces(ctx, serverID, items); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "保存命名空间失败", err)
	}
	invalidateReasoningGraphs(ctx)
	return s.listNamespaces(ctx, serverID)
}

// ==================== 市场 ====================

func (s *McpServerService) GetMarket(ctx context.Context) ([]mcpMarketPreset, error) {
	items := make([]mcpMarketPreset, 0, len(marketPresets))
	for _, preset := range marketPresets {
		server, err := s.repo.GetServerByName(ctx, preset.Name, false)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询市场预设失败", err)
		}
		items = append(items, mcpMarketPreset{
			PresetID:       preset.PresetID,
			Name:           preset.Name,
			Description:    preset.Description,
			CapabilityTags: preset.CapabilityTags,
			Installed:      server != nil,
		})
	}
	return items, nil
}

// ==================== 调用审计 ====================

func (s *McpServerService) ListCalls(ctx context.Context, q *bo.McpCallQuery) (*vo.PageResult[vo.McpCallVO], error) {
	page, size := q.PageNum, q.PageSize
	calls, total, err := s.repo.PaginateCalls(ctx, page, size, q.ServerID, q.ToolName)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询调用审计失败", err)
	}
	items := make([]vo.McpCallVO, 0, len(calls))
	for i := range calls {
		items = append(items, vo.McpCallVO{
			ID:         calls[i].ID,
			UserID:     calls[i].UserID,
			ServerID:   calls[i].ServerID,
			ServerName: calls[i].ServerName,
			ToolName:   calls[i].ToolName,
			Result:     calls[i].Result,
			LatencyMs:  calls[i].LatencyMs,
			CreateTime: calls[i].CreateTime,
		})
	}
	return &vo.PageResult[vo.McpCallVO]{List: items, Total: total}, nil
}

// ==================== 内部方法 ====================

func (s *McpServerService) requireServer(ctx context.Context, serverID int64) (*model.SysAiMcpServer, error) {
	server, err := s.repo.GetServer(ctx, serverID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 MCP Server 失败", err)
	}
	if server == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "MCP Server 不存在")
	}
	return server, nil
}

func (s *McpServerService) listNamespaces(ctx context.Context, serverID int64) ([]vo.McpNamespaceVO, error) {
	rows, err := s.repo.ListNamespaces(ctx, serverID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询命名空间失败", err)
	}
	items := make([]vo.McpNamespaceVO, 0, len(rows))
	for i := range rows {
		var toolNames []string
		if len(rows[i].ToolNames) > 0 {
			_ = json.Unmarshal(rows[i].ToolNames, &toolNames)
		}
		items = append(items, vo.McpNamespaceVO{Name: rows[i].Namespace, ToolNames: nonNilStrings(toolNames)})
	}
	return items, nil
}

func (s *McpServerService) knownToolNames(ctx context.Context, serverID int64) (map[string]bool, error) {
	var names []string
	err := s.db.WithContext(ctx).Table("sys_ai_mcp_tool").
		Where("server_id = ?", serverID).Pluck("name", &names).Error
	if err != nil {
		return nil, err
	}
	result := make(map[string]bool, len(names))
	for _, name := range names {
		result[name] = true
	}
	return result, nil
}

// countAgentReferences 统计关联了该 Server 命名空间的 Agent 数（同名命名空间跨 Server 复用时保守计入）
func (s *McpServerService) countAgentReferences(ctx context.Context, serverID int64) (int64, error) {
	var count int64
	err := s.db.WithContext(ctx).Table("sys_ai_agent_mcp AS am").
		Joins("JOIN sys_ai_mcp_namespace AS ns ON ns.namespace = am.mcp_namespace").
		Where("ns.server_id = ?", serverID).
		Distinct("am.agent_id").Count(&count).Error
	return count, err
}

func toMcpServerVO(s *model.SysAiMcpServer) vo.McpServerVO {
	credentialConfigured := len(s.Credentials) > 0 && string(s.Credentials) != "null" && string(s.Credentials) != "{}"
	return vo.McpServerVO{
		ID:                   s.ID,
		Name:                 s.Name,
		Description:          s.Description,
		ProtocolType:         s.ProtocolType,
		Endpoint:             s.Endpoint,
		AuthType:             s.AuthType,
		Status:               s.Status,
		Health:               s.Health,
		LastCheckTime:        s.LastCheckTime,
		ToolCount:            s.ToolCount,
		CredentialConfigured: credentialConfigured,
		CreateTime:           s.CreatedAt,
		UpdateTime:           s.UpdatedAt,
	}
}

func nonNilStrings(values []string) []string {
	if values == nil {
		return []string{}
	}
	return values
}
