package aicontract

import (
	"encoding/json"
	"sort"
	"strings"
	"time"
)

// 本文件是契约之外的手写辅助：库中 JSON 列与契约 DTO 之间的解析/序列化，以及从 blocks
// 派生纯文本。生成器只产出 contract.gen.go，不改动本文件。
//
// 契约要求 blocks/artifactIds/memoryIds/suggestions 为数组，nil 切片会被序列化成 null，
// 故所有解析函数在"无值"时返回空切片而非 nil。

// 时间列可能由三端分别写入：RFC3339 带时区，或本地时间无时区。
// 解析顺序从严格到宽松，保证紧凑格式（不带 Z/偏移）也能判定中断是否过期。
// withZone 标记 layout 自带 Z/偏移：带时区的按字面时区解析，无时区的按本地时区解析。
var interruptTimeLayouts = []struct {
	layout   string
	withZone bool
}{
	{time.RFC3339Nano, true},
	{time.RFC3339, true},
	{"2006-01-02T15:04:05", false},
	{"2006-01-02 15:04:05", false},
}

// DecodeBlocks 解析 blocks JSON 列：空值、JSON null、非法文本一律返回空切片。
// 返回结果按 seq 升序（稳定排序），保证同一消息的块序在会话列表/详情/导出三处一致。
func DecodeBlocks(raw string) []Block {
	if strings.TrimSpace(raw) == "" {
		return []Block{}
	}
	var blocks []Block
	if err := json.Unmarshal([]byte(raw), &blocks); err != nil || blocks == nil {
		return []Block{}
	}
	sort.SliceStable(blocks, func(i, j int) bool { return blocks[i].Seq < blocks[j].Seq })
	return blocks
}

// EncodeBlocks 序列化 blocks 为库中 JSON 文本；空集合写 "[]"（blocks 列为 NOT NULL）。
func EncodeBlocks(blocks []Block) string {
	if len(blocks) == 0 {
		return "[]"
	}
	payload, err := json.Marshal(blocks)
	if err != nil {
		return "[]"
	}
	return string(payload)
}

// DeriveText 从 blocks 派生纯文本：只取 kind=text 的块，按 seq 升序、块间空行拼接。
// 思考与工具块不进入正文——它们由前端按分区各自渲染，导出与审计文本统一走本函数。
func DeriveText(blocks []Block) string {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Kind != BlockKindText || block.Text == nil || *block.Text == "" {
			continue
		}
		parts = append(parts, *block.Text)
	}
	return strings.Join(parts, "\n\n")
}

// DecodePlan 解析 plan JSON 列（无计划或非法返回 nil，契约允许 plan 为 null）。
// tasks/revisions 为契约必填数组，缺失时补空切片而不是下发 null。
func DecodePlan(raw string) *Plan {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var plan Plan
	if err := json.Unmarshal([]byte(raw), &plan); err != nil {
		return nil
	}
	if plan.Tasks == nil {
		plan.Tasks = []PlanTask{}
	}
	if plan.Revisions == nil {
		plan.Revisions = []PlanRevision{}
	}
	return &plan
}

// DecodeInterrupt 解析 pending_interrupt JSON 列（无中断或非法返回 nil）。
func DecodeInterrupt(raw string) *Interrupt {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var interrupt Interrupt
	if err := json.Unmarshal([]byte(raw), &interrupt); err != nil {
		return nil
	}
	return &interrupt
}

// NormalizeInterrupt 按 expiresAt 重算可续性：过期的中断 actionable 置 false，
// 明确标注"不可再 resume"，而不是把过期中断当成可执行。expiresAt 不可解析时按过期处理。
func NormalizeInterrupt(interrupt *Interrupt, now time.Time) *Interrupt {
	if interrupt == nil {
		return nil
	}
	normalized := *interrupt
	expiresAt, ok := parseInterruptTime(interrupt.ExpiresAt)
	normalized.Actionable = interrupt.Actionable && ok && now.Before(expiresAt)
	return &normalized
}

// DecodeSuggestions 解析 suggestions JSON 列（string[]，无值返回空切片）。
func DecodeSuggestions(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return []string{}
	}
	var suggestions []string
	if err := json.Unmarshal([]byte(raw), &suggestions); err != nil || suggestions == nil {
		return []string{}
	}
	return suggestions
}

// DecodeIDList 解析整型数组 JSON 列（used_memory_ids 等，无值返回空切片）。
func DecodeIDList(raw string) []int64 {
	if strings.TrimSpace(raw) == "" {
		return []int64{}
	}
	var ids []int64
	if err := json.Unmarshal([]byte(raw), &ids); err != nil || ids == nil {
		return []int64{}
	}
	return ids
}

// DecodeSubAgentUsage 解析 sub_agent_usage JSON 列（SubAgentUsage[]，无值返回 nil，契约允许 null）。
func DecodeSubAgentUsage(raw string) []SubAgentUsage {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var usage []SubAgentUsage
	if err := json.Unmarshal([]byte(raw), &usage); err != nil || len(usage) == 0 {
		return nil
	}
	return usage
}

// parseInterruptTime 按 layout 是否自带时区选择解析方式：带 Z/offset 的按字面时区解析，
// 无时区的按本地时区解析——三端写入的紧凑格式是本地时间，k8s UTC 容器里若按 UTC 解析，
// 过期判定会与真实时刻相差一个时区偏移（上海 8 小时）。全部失败返回 false。
func parseInterruptTime(value string) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, false
	}
	for _, item := range interruptTimeLayouts {
		var (
			parsed time.Time
			err    error
		)
		if item.withZone {
			parsed, err = time.Parse(item.layout, value)
		} else {
			parsed, err = time.ParseInLocation(item.layout, value, time.Local)
		}
		if err == nil {
			return parsed, true
		}
	}
	return time.Time{}, false
}
