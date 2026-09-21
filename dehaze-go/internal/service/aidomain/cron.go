package aidomain

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	// 内嵌 IANA 时区数据库：任务时区（如 Asia/Shanghai）不依赖宿主 zoneinfo 是否存在
	_ "time/tzdata"

	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// cronField 单字段的取值集合。
type cronField struct {
	values map[int]struct{}
	star   bool
}

// cronSchedule 解析后的 5 位 Cron 表达式。
type cronSchedule struct {
	minute  cronField
	hour    cronField
	day     cronField
	month   cronField
	weekday cronField
}

var cronAliases = map[string]string{
	"@hourly":   "0 * * * *",
	"@daily":    "0 0 * * *",
	"@midnight": "0 0 * * *",
	"@weekly":   "0 0 * * 0",
	"@monthly":  "0 0 1 * *",
	"@yearly":   "0 0 1 1 *",
	"@annually": "0 0 1 1 *",
}

// normalizeCron 归一化触发规则为标准 5 位 Cron（对齐 python 的 daily/weekly/monthly 频率标识）。
func normalizeCron(raw string) (string, error) {
	text := strings.TrimSpace(raw)
	if !strings.Contains(text, "@") || strings.HasPrefix(text, "@") {
		return text, nil
	}
	parts := strings.Split(text, "@")
	if len(parts) < 2 || len(parts) > 3 {
		return text, nil
	}
	switch parts[0] {
	case "daily", "weekly", "monthly":
	default:
		return text, nil
	}
	hour, minute, err := parseClock(parts[len(parts)-1])
	if err != nil {
		return "", err
	}
	if parts[0] == "daily" {
		return fmt.Sprintf("%d %d * * *", minute, hour), nil
	}
	day := strings.ToLower(strings.TrimSpace(parts[1]))
	if parts[0] == "weekly" {
		weekday, ok := weekdayAlias(day)
		if !ok {
			return text, nil
		}
		return fmt.Sprintf("%d %d * * %d", minute, hour, weekday), nil
	}
	dom, err := strconv.Atoi(day)
	if err != nil {
		return text, nil
	}
	if dom < 1 || dom > 31 {
		return "", common.NewBizError(common.PARAM_ERROR, "触发规则日期超出范围: "+raw)
	}
	return fmt.Sprintf("%d %d %d * *", minute, hour, dom), nil
}

func parseClock(raw string) (hour, minute int, err error) {
	pieces := strings.Split(raw, ":")
	if len(pieces) != 2 {
		return 0, 0, common.NewBizError(common.PARAM_ERROR, "触发规则时间格式非法: "+raw)
	}
	hour, errHour := strconv.Atoi(strings.TrimSpace(pieces[0]))
	minute, errMinute := strconv.Atoi(strings.TrimSpace(pieces[1]))
	if errHour != nil || errMinute != nil {
		return 0, 0, common.NewBizError(common.PARAM_ERROR, "触发规则时间格式非法: "+raw)
	}
	if hour < 0 || hour > 23 || minute < 0 || minute > 59 {
		return 0, 0, common.NewBizError(common.PARAM_ERROR, "触发规则时间超出范围: "+raw)
	}
	return hour, minute, nil
}

// weekdayAlias 星期别名（对齐 cron 语义：0 与 7 均为周日）。
func weekdayAlias(day string) (int, bool) {
	switch day {
	case "mon":
		return 1, true
	case "tue":
		return 2, true
	case "wed":
		return 3, true
	case "thu":
		return 4, true
	case "fri":
		return 5, true
	case "sat":
		return 6, true
	case "sun":
		return 0, true
	}
	value, err := strconv.Atoi(day)
	if err != nil {
		return 0, false
	}
	return value % 7, true
}

// parseCron 解析 5 位 Cron 表达式，非法时返回参数错误。
func parseCron(raw string) (*cronSchedule, error) {
	text := strings.TrimSpace(raw)
	if alias, ok := cronAliases[text]; ok {
		text = alias
	}
	fields := strings.Fields(text)
	if len(fields) != 5 {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	minute, err := parseCronField(fields[0], 0, 59)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	hour, err := parseCronField(fields[1], 0, 23)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	day, err := parseCronField(fields[2], 1, 31)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	month, err := parseCronField(fields[3], 1, 12)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	weekday, err := parseCronField(fields[4], 0, 7)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "Cron 表达式非法: "+raw)
	}
	// cron 语义中 7 与 0 同为周日
	if _, ok := weekday.values[7]; ok {
		weekday.values[0] = struct{}{}
	}
	return &cronSchedule{minute: minute, hour: hour, day: day, month: month, weekday: weekday}, nil
}

func parseCronField(raw string, min, max int) (cronField, error) {
	field := cronField{values: map[int]struct{}{}}
	raw = strings.TrimSpace(raw)
	if raw == "*" {
		field.star = true
		for value := min; value <= max; value++ {
			field.values[value] = struct{}{}
		}
		return field, nil
	}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			return field, fmt.Errorf("empty segment")
		}
		step := 1
		body := part
		if idx := strings.Index(part, "/"); idx >= 0 {
			body = part[:idx]
			parsed, err := strconv.Atoi(part[idx+1:])
			if err != nil || parsed <= 0 {
				return field, fmt.Errorf("invalid step")
			}
			step = parsed
		}
		start, end := min, max
		if body != "*" {
			if idx := strings.Index(body, "-"); idx >= 0 {
				low, errLow := strconv.Atoi(body[:idx])
				high, errHigh := strconv.Atoi(body[idx+1:])
				if errLow != nil || errHigh != nil {
					return field, fmt.Errorf("invalid range")
				}
				start, end = low, high
			} else {
				value, err := strconv.Atoi(body)
				if err != nil {
					return field, fmt.Errorf("invalid value")
				}
				start, end = value, value
				if step > 1 {
					end = max
				}
			}
		}
		if start < min || end > max || start > end {
			return field, fmt.Errorf("out of range")
		}
		for value := start; value <= end; value += step {
			field.values[value] = struct{}{}
		}
	}
	return field, nil
}

func (s *cronSchedule) matches(t time.Time) bool {
	if _, ok := s.minute.values[t.Minute()]; !ok {
		return false
	}
	if _, ok := s.hour.values[t.Hour()]; !ok {
		return false
	}
	if _, ok := s.month.values[int(t.Month())]; !ok {
		return false
	}
	// 标准 cron 语义：日与星期为 OR（任一命中即触发），但两者都为 * 时按日判定
	weekday := int(t.Weekday())
	if _, ok := s.weekday.values[weekday]; !ok && !s.weekday.star {
		if _, dayOk := s.day.values[t.Day()]; !dayOk || s.day.star {
			return false
		}
	}
	// 星期命中时仍需日字段约束生效（当两者均非 * 时任一命中即可）
	if s.day.star || s.weekday.star {
		if _, ok := s.day.values[t.Day()]; !ok && !s.day.star {
			return false
		}
	}
	return true
}

// nextTimes 返回从 after 之后开始的 count 次触发时间（按指定时区）。
func (s *cronSchedule) nextTimes(after time.Time, count int) []time.Time {
	result := make([]time.Time, 0, count)
	current := after.Truncate(time.Minute).Add(time.Minute)
	limit := after.AddDate(5, 0, 0)
	for len(result) < count && current.Before(limit) {
		if s.matches(current) {
			result = append(result, current)
		}
		current = current.Add(time.Minute)
	}
	return result
}

// resolveLocation 解析任务时区。
func resolveLocation(timezone string) (*time.Location, error) {
	if timezone == "" {
		timezone = "Asia/Shanghai"
	}
	location, err := time.LoadLocation(timezone)
	if err != nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "任务时区非法: "+timezone)
	}
	return location, nil
}

// computeNextTrigger 计算下一次触发时间（按任务时区，返回该时区本地时间）。
func computeNextTrigger(cron, timezone string) (time.Time, error) {
	schedule, err := parseCron(cron)
	if err != nil {
		return time.Time{}, err
	}
	location, err := resolveLocation(timezone)
	if err != nil {
		return time.Time{}, err
	}
	times := schedule.nextTimes(time.Now().In(location), 1)
	if len(times) == 0 {
		return time.Time{}, common.NewBizError(common.PARAM_ERROR, "Cron 表达式无法解析出触发时间")
	}
	return times[0].In(location), nil
}

// describeCron Cron 的人类可读描述（对齐 python 的简版映射）。
func describeCron(cron string) string {
	parts := strings.Fields(cron)
	if len(parts) != 5 {
		return cron
	}
	minute, hour, day, month, weekday := parts[0], parts[1], parts[2], parts[3], parts[4]
	weekdayNames := map[string]string{
		"0": "日", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "日",
	}
	if minute == "*" && hour == "*" && day == "*" && month == "*" && weekday == "*" {
		return "每分钟"
	}
	if hour == "*" && day == "*" && month == "*" && weekday == "*" {
		if minute == "0" {
			return "每小时整点"
		}
		return "每小时 " + formatMinute(minute) + "分"
	}
	if hour != "*" && day == "*" && month == "*" && weekday == "*" {
		return "每天 " + zeroPad(hour) + "点" + formatMinute(minute) + "分"
	}
	if day == "*" && month == "*" && weekday != "*" {
		names := []string{}
		for _, item := range strings.Split(weekday, ",") {
			if name, ok := weekdayNames[item]; ok {
				names = append(names, "周"+name)
			}
		}
		return "每周" + strings.Join(names, "、") + " " + describeTimePart(hour, minute)
	}
	if month == "*" && weekday == "*" && day != "*" {
		days := []string{}
		for _, item := range strings.Split(day, ",") {
			days = append(days, item+"号")
		}
		return "每月" + strings.Join(days, "、") + " " + describeTimePart(hour, minute)
	}
	if month != "*" && weekday == "*" && day != "*" {
		months := []string{}
		for _, item := range strings.Split(month, ",") {
			months = append(months, item+"月")
		}
		days := []string{}
		for _, item := range strings.Split(day, ",") {
			days = append(days, item+"号")
		}
		return "每年" + strings.Join(months, "、") + strings.Join(days, "、") +
			" " + zeroPad(hour) + "点" + formatMinute(minute) + "分"
	}
	return fmt.Sprintf("Cron(%s)", cron)
}

func describeTimePart(hour, minute string) string {
	if hour != "*" {
		return zeroPad(hour) + "点" + formatMinute(minute) + "分"
	}
	return "每小时" + formatMinute(minute) + "分"
}

func zeroPad(value string) string {
	if len(value) == 1 {
		return "0" + value
	}
	return value
}

func formatMinute(minute string) string {
	if minute == "*" {
		return "00"
	}
	return zeroPad(minute)
}
