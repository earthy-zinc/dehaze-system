package aidomain

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// AgentVersionVO 版本历史项。
type AgentVersionVO struct {
	ID         int64  `json:"id"`
	AgentID    int64  `json:"agentId"`
	VersionNo  int    `json:"versionNo"`
	Status     int    `json:"status"`
	ChangeNote string `json:"changeNote,omitempty"`
	OperatorID *int64 `json:"operatorId,omitempty"`
	CreateTime string `json:"createTime,omitempty"`
}

// AgentVersionDetailVO 版本快照详情。
type AgentVersionDetailVO struct {
	AgentVersionVO
	Snapshot json.RawMessage `json:"snapshot"`
}

// ListVersions 版本历史分页。
func (s *AgentVersionService) ListVersions(ctx context.Context, agentID int64, page, size int) (*vo.PageResult[AgentVersionVO], error) {
	items, total, err := s.agents.ListVersions(ctx, agentID, (page-1)*size, size)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询版本历史失败", err)
	}
	list := make([]AgentVersionVO, 0, len(items))
	for i := range items {
		list = append(list, *toVersionVO(&items[i]))
	}
	return &vo.PageResult[AgentVersionVO]{List: list, Total: total}, nil
}

// GetVersionDetail 版本快照详情。
func (s *AgentVersionService) GetVersionDetail(ctx context.Context, agentID int64, versionNo int) (*AgentVersionDetailVO, error) {
	version, err := s.agents.GetVersion(ctx, agentID, versionNo)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询版本失败", err)
	}
	if version == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "版本快照不存在")
	}
	return &AgentVersionDetailVO{
		AgentVersionVO: *toVersionVO(version),
		Snapshot:       rawJSON(version.Snapshot),
	}, nil
}

// DiffVersions 版本差异对比（返回叶节点差异列表）。
func (s *AgentVersionService) DiffVersions(ctx context.Context, agentID int64, baseVersionNo, targetVersionNo int) ([]map[string]any, error) {
	load := func(versionNo int) (map[string]any, error) {
		version, err := s.agents.GetVersion(ctx, agentID, versionNo)
		if err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询版本失败", err)
		}
		if version == nil {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, fmt.Sprintf("版本 %d 不存在", versionNo))
		}
		var snapshot map[string]any
		if err := json.Unmarshal([]byte(version.Snapshot), &snapshot); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "解析版本快照失败", err)
		}
		return snapshot, nil
	}
	base, err := load(baseVersionNo)
	if err != nil {
		return nil, err
	}
	target, err := load(targetVersionNo)
	if err != nil {
		return nil, err
	}
	diffs := []map[string]any{}
	diffSnapshot(base, target, "", &diffs)
	return diffs, nil
}

// diffSnapshot 递归比较两个快照，仅记录叶节点差异（列表按整体比较）。
func diffSnapshot(base, target any, prefix string, acc *[]map[string]any) {
	baseMap, baseIsMap := base.(map[string]any)
	targetMap, targetIsMap := target.(map[string]any)
	if baseIsMap && targetIsMap {
		keys := make([]string, 0, len(baseMap)+len(targetMap))
		seen := map[string]struct{}{}
		for key := range baseMap {
			keys = append(keys, key)
			seen[key] = struct{}{}
		}
		for key := range targetMap {
			if _, ok := seen[key]; !ok {
				keys = append(keys, key)
			}
		}
		sort.Strings(keys)
		for _, key := range keys {
			path := key
			if prefix != "" {
				path = prefix + "." + key
			}
			diffSnapshot(baseMap[key], targetMap[key], path, acc)
		}
		return
	}
	if !equalJSON(base, target) {
		*acc = append(*acc, map[string]any{"field": prefix, "base": base, "target": target})
	}
}

func equalJSON(a, b any) bool {
	left, errLeft := json.Marshal(a)
	right, errRight := json.Marshal(b)
	if errLeft != nil || errRight != nil {
		return false
	}
	return string(left) == string(right)
}

func toVersionVO(version *model.SysAiAgentVersion) *AgentVersionVO {
	return &AgentVersionVO{
		ID:         version.ID,
		AgentID:    version.AgentID,
		VersionNo:  version.VersionNo,
		Status:     version.Status,
		ChangeNote: derefString(version.ChangeNote),
		OperatorID: version.OperatorID,
		CreateTime: formatTime(version.CreateTime),
	}
}

func applySnapshotString(fields map[string]any, snapshot map[string]any, snapshotKey, column string) {
	value, ok := snapshot[snapshotKey]
	if !ok {
		return
	}
	if value == nil {
		fields[column] = nil
		return
	}
	if text, ok := value.(string); ok {
		fields[column] = text
		return
	}
	fields[column] = fmt.Sprint(value)
}

func snapshotStrings(snapshot map[string]any, key string) []string {
	raw, ok := snapshot[key].([]any)
	if !ok {
		return []string{}
	}
	result := make([]string, 0, len(raw))
	for _, item := range raw {
		if text, ok := item.(string); ok {
			result = append(result, text)
		}
	}
	return result
}

func snapshotSubagents(snapshot map[string]any) []model.SysAiAgentSubagent {
	raw, ok := snapshot["subagents"].([]any)
	if !ok {
		return []model.SysAiAgentSubagent{}
	}
	result := make([]model.SysAiAgentSubagent, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		link := model.SysAiAgentSubagent{}
		if id, ok := entry["agent_id"].(float64); ok {
			link.SubagentAgentID = int64(id)
		}
		if priority, ok := entry["priority"].(float64); ok {
			link.Priority = int(priority)
		}
		if endpoint, ok := entry["endpoint_id"].(float64); ok {
			endpointID := int64(endpoint)
			link.EndpointID = &endpointID
		}
		result = append(result, link)
	}
	return result
}

func splitDotted(key string) []string {
	parts := []string{}
	current := ""
	for _, ch := range key {
		if ch == '.' {
			parts = append(parts, current)
			current = ""
			continue
		}
		current += string(ch)
	}
	return append(parts, current)
}

// coerceScalar 字典值字符串转 int/float/bool，无法转换保留字符串。
func coerceScalar(raw string) any {
	switch raw {
	case "true":
		return true
	case "false":
		return false
	}
	var intValue int
	if _, err := fmt.Sscanf(raw, "%d", &intValue); err == nil && fmt.Sprintf("%d", intValue) == raw {
		return intValue
	}
	var floatValue float64
	if _, err := fmt.Sscanf(raw, "%g", &floatValue); err == nil && fmt.Sprintf("%g", floatValue) == raw {
		return floatValue
	}
	return raw
}
