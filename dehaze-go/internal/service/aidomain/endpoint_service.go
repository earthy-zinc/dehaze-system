package aidomain

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

var validAuthTypes = map[string]struct{}{
	"apiKey": {}, "http": {}, "oauth2": {}, "openIdConnect": {}, "mutualTLS": {},
}

// blockedIPv4Prefixes 标准库 IsPrivate/IsLoopback/IsLinkLocalUnicast 未覆盖的 IPv4 段，
// 对齐 python ipaddress.is_private 的覆盖范围：
//   - 安全策略额外封禁的保留段 0/8、9/8、11/8、21/8、30/8；
//   - 运营商级 NAT 100.64/10；
//   - 组播与保留 224/4、240/4；
//   - 文档段 TEST-NET-1/2/3 与基准测试段 198.18/15。
var blockedIPv4Prefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("9.0.0.0/8"),
	netip.MustParsePrefix("11.0.0.0/8"),
	netip.MustParsePrefix("21.0.0.0/8"),
	netip.MustParsePrefix("30.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"),
	netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"),
	netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("224.0.0.0/4"),
	netip.MustParsePrefix("240.0.0.0/4"),
}

// EndpointVO 外部 A2A 端点响应。
type EndpointVO struct {
	ID           int64           `json:"id"`
	Name         string          `json:"name"`
	AgentCardURL string          `json:"agentCardUrl,omitempty"`
	BaseURL      string          `json:"baseUrl"`
	AuthType     string          `json:"authType"`
	AgentCard    json.RawMessage `json:"agentCard,omitempty"`
	Status       int             `json:"status"`
	CreateTime   string          `json:"createTime,omitempty"`
}

// EndpointCreateForm 注册端点表单。
type EndpointCreateForm struct {
	Name         string  `json:"name"`
	AgentCardURL *string `json:"agentCardUrl"`
	BaseURL      string  `json:"baseUrl"`
	AuthType     string  `json:"authType"`
	Credential   *string `json:"credential"`
	Status       *int    `json:"status" binding:"omitempty,oneof=0 1"`
}

// EndpointUpdateForm 更新端点表单。
type EndpointUpdateForm struct {
	Name         *string `json:"name"`
	AgentCardURL *string `json:"agentCardUrl"`
	AuthType     *string `json:"authType"`
	Credential   *string `json:"credential"`
	Status       *int    `json:"status" binding:"omitempty,oneof=0 1"`
}

// EndpointService 外部 A2A 端点管理（含 Agent Card 拉取缓存）。
type EndpointService struct {
	agents   *repo.AgentRepository
	auditLog *auditlogservice.AuditLogService
	client   *http.Client
}

// NewEndpointService 构造 EndpointService。
func NewEndpointService(agents *repo.AgentRepository, auditLog *auditlogservice.AuditLogService) *EndpointService {
	return &EndpointService{
		agents:   agents,
		auditLog: auditLog,
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

// Create 注册外部 A2A 端点（唯一键软删行复活）。
func (s *EndpointService) Create(ctx context.Context, form *EndpointCreateForm) (*EndpointVO, error) {
	if strings.TrimSpace(form.Name) == "" || strings.TrimSpace(form.BaseURL) == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "端点名称与地址不能为空")
	}
	authType := form.AuthType
	if authType == "" {
		authType = "http"
	}
	if _, ok := validAuthTypes[authType]; !ok {
		return nil, common.NewBizError(common.PARAM_ERROR, "认证方式取值非法")
	}
	baseURL := strings.TrimRight(form.BaseURL, "/")
	if !isSafeURL(baseURL) || (form.AgentCardURL != nil && !isSafeURL(*form.AgentCardURL)) {
		return nil, common.NewBizError(common.PARAM_ERROR, "base_url/agent_card_url 仅支持 https 且禁止内网地址")
	}
	status := 1
	if form.Status != nil {
		status = *form.Status
	}

	existing, err := s.agents.GetEndpointByBaseURL(ctx, baseURL)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点失败", err)
	}
	var endpointID int64
	if existing != nil {
		if existing.Deleted == 0 {
			return nil, common.NewBizError(common.DATA_EXISTS, "该端点地址已注册")
		}
		// 软删行占用唯一键 base_url，复活原行并覆盖为新表单值
		fields := map[string]any{
			"name":           form.Name,
			"agent_card_url": form.AgentCardURL,
			"auth_type":      authType,
			"credential":     form.Credential,
			"status":         status,
		}
		if err := s.agents.ReviveEndpoint(ctx, existing.ID, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复端点失败", err)
		}
		endpointID = existing.ID
	} else {
		endpoint := &model.SysAiAgentEndpoint{
			Name:         form.Name,
			AgentCardURL: form.AgentCardURL,
			BaseURL:      baseURL,
			AuthType:     authType,
			Credential:   form.Credential,
			Status:       status,
		}
		if err := s.agents.CreateEndpoint(ctx, endpoint); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "注册端点失败", err)
		}
		endpointID = endpoint.ID
	}
	// 注册成功后拉取 Agent Card（失败不阻断创建，仅告警）
	s.refreshAgentCard(ctx, endpointID)
	saved, err := s.agents.GetEndpoint(ctx, endpointID)
	if err != nil || saved == nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点失败", err)
	}
	return toEndpointVO(saved), nil
}

// Update 更新端点。
func (s *EndpointService) Update(ctx context.Context, id int64, form *EndpointUpdateForm) (*EndpointVO, error) {
	endpoint, err := s.agents.GetEndpoint(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点失败", err)
	}
	if endpoint == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "端点不存在")
	}
	fields := map[string]any{}
	if form.Name != nil {
		fields["name"] = *form.Name
	}
	if form.AgentCardURL != nil {
		if !isSafeURL(*form.AgentCardURL) {
			return nil, common.NewBizError(common.PARAM_ERROR, "agent_card_url 仅支持 https 且禁止内网地址")
		}
		fields["agent_card_url"] = *form.AgentCardURL
	}
	if form.AuthType != nil {
		if _, ok := validAuthTypes[*form.AuthType]; !ok {
			return nil, common.NewBizError(common.PARAM_ERROR, "认证方式取值非法")
		}
		fields["auth_type"] = *form.AuthType
	}
	if form.Credential != nil {
		fields["credential"] = *form.Credential
	}
	if form.Status != nil {
		fields["status"] = *form.Status
	}
	if len(fields) > 0 {
		if err := s.agents.UpdateEndpointFields(ctx, id, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新端点失败", err)
		}
	}
	s.refreshAgentCard(ctx, id)
	saved, err := s.agents.GetEndpoint(ctx, id)
	if err != nil || saved == nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点失败", err)
	}
	return toEndpointVO(saved), nil
}

// Delete 删除端点（软删 + 审计）。
func (s *EndpointService) Delete(ctx context.Context, id, operatorID int64) error {
	endpoint, err := s.agents.GetEndpoint(ctx, id)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点失败", err)
	}
	if endpoint == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "端点不存在")
	}
	if err := s.agents.SoftDeleteEndpoints(ctx, []int64{id}, operatorID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除端点失败", err)
	}
	if s.auditLog != nil {
		s.auditLog.RecordAuditAsync(ctx, operatorID, "ai_agent_endpoint", id, "delete", "ai_agent",
			map[string]any{"name": endpoint.Name, "base_url": endpoint.BaseURL}, nil, "", "")
	}
	return nil
}

// List 端点分页列表。
func (s *EndpointService) List(ctx context.Context, page, size int, keyword string, status *int) (*vo.PageResult[EndpointVO], error) {
	items, total, err := s.agents.PaginateEndpoints(ctx, page, size, keyword, status)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询端点列表失败", err)
	}
	result := make([]EndpointVO, 0, len(items))
	for i := range items {
		result = append(result, *toEndpointVO(&items[i]))
	}
	return &vo.PageResult[EndpointVO]{List: result, Total: total}, nil
}

// refreshAgentCard 拉取并缓存 Agent Card（拉取失败仅告警）。
func (s *EndpointService) refreshAgentCard(ctx context.Context, endpointID int64) {
	endpoint, err := s.agents.GetEndpoint(ctx, endpointID)
	if err != nil || endpoint == nil || endpoint.AgentCardURL == nil || *endpoint.AgentCardURL == "" {
		return
	}
	cardURL := *endpoint.AgentCardURL
	if !isSafeURL(cardURL) {
		logger.Warn("Agent Card 地址不安全，跳过拉取", zap.String("url", cardURL))
		return
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, cardURL, nil)
	if err != nil {
		return
	}
	response, err := s.client.Do(request)
	if err != nil {
		logger.Warn("Agent Card 拉取失败", zap.Int64("endpointId", endpointID), zap.Error(err))
		return
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		logger.Warn("Agent Card 拉取异常状态", zap.Int64("endpointId", endpointID), zap.Int("status", response.StatusCode))
		return
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return
	}
	var card map[string]any
	if err := json.Unmarshal(body, &card); err != nil {
		logger.Warn("Agent Card 解析失败", zap.Int64("endpointId", endpointID), zap.Error(err))
		return
	}
	if err := s.agents.UpdateEndpointFields(ctx, endpointID, map[string]any{"agent_card": string(body)}); err != nil {
		logger.Warn("Agent Card 缓存写入失败", zap.Int64("endpointId", endpointID), zap.Error(err))
	}
}

// isSafeURL SSRF 防护：仅允许 https，主机（含域名解析出的全部地址）不得为环回/内网/链路本地/保留地址。
//
// 对齐 python app/utils/ssrf.is_safe_url：域名按全部解析结果判定（防 DNS 重绑定）、解析失败保守拒绝。
// 字面 IP 一律走标准库语义判定，不再按字符串前缀猜段——前缀匹配既漏 IPv6（fe80::/10、::ffff: 映射）
// 又漏 IPv4 段（如 100.64/10）。
func isSafeURL(raw string) bool {
	if strings.TrimSpace(raw) == "" {
		return false
	}
	parsed, err := url.Parse(raw)
	if err != nil || !strings.EqualFold(parsed.Scheme, "https") {
		return false
	}
	host := parsed.Hostname()
	if host == "" {
		return false
	}
	lowered := strings.ToLower(host)
	if lowered == "localhost" || strings.HasSuffix(lowered, ".local") || strings.HasSuffix(lowered, ".internal") {
		return false
	}
	if ip := net.ParseIP(lowered); ip != nil {
		return !isInternalIP(ip)
	}
	addresses, err := net.LookupIP(lowered)
	if err != nil || len(addresses) == 0 {
		return false
	}
	for _, address := range addresses {
		if isInternalIP(address) {
			return false
		}
	}
	return true
}

// isInternalIP 判定地址是否为环回/内网/链路本地/未指定/组播/保留。
//
// Go 的 IsPrivate/IsLoopback/IsLinkLocalUnicast 内部先做 To4()，故 ::ffff:127.0.0.1 这类
// IPv4 映射地址按映射到的 IPv4 判定（与 python ipaddress 语义一致）。
func isInternalIP(ip net.IP) bool {
	// 环回/私有(RFC1918 + ULA fc00::/7)/链路本地(169.254、fe80::/10)/未指定/组播
	if ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() || ip.IsMulticast() || ip.IsInterfaceLocalMulticast() || ip.IsPrivate() {
		return true
	}
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true
	}
	// Unmap 让 ::ffff:127.0.0.1 这类映射地址也参与 IPv4 段判定
	addr = addr.Unmap()
	for _, prefix := range blockedIPv4Prefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

func toEndpointVO(endpoint *model.SysAiAgentEndpoint) *EndpointVO {
	return &EndpointVO{
		ID:           endpoint.ID,
		Name:         endpoint.Name,
		AgentCardURL: derefString(endpoint.AgentCardURL),
		BaseURL:      endpoint.BaseURL,
		AuthType:     endpoint.AuthType,
		AgentCard:    rawJSON(endpoint.AgentCard),
		Status:       endpoint.Status,
		CreateTime:   formatTime(endpoint.CreateTime),
	}
}
