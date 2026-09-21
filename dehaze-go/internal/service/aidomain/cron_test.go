package aidomain

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeCron(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{name: "标准表达式原样", input: "0 9 * * *", want: "0 9 * * *"},
		{name: "daily 频率标识", input: "daily@09:30", want: "30 9 * * *"},
		{name: "weekly 星期别名", input: "weekly@mon@08:00", want: "0 8 * * 1"},
		{name: "weekly 数字日", input: "weekly@6@23:59", want: "59 23 * * 6"},
		{name: "monthly 指定日", input: "monthly@15@07:05", want: "5 7 15 * *"},
		{name: "非法时间格式", input: "daily@25:00", wantErr: true},
		{name: "非法日期", input: "monthly@40@10:00", wantErr: true},
		{name: "无法识别原样返回", input: "yearly@01:00", want: "yearly@01:00"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeCron(tc.input)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestParseCronRejectsInvalidExpression(t *testing.T) {
	for _, raw := range []string{"", "0 9 * *", "60 9 * * *", "0 25 * * *", "0 9 0 * *", "a b c d e"} {
		t.Run(raw, func(t *testing.T) {
			_, err := parseCron(raw)
			require.Error(t, err)
		})
	}
}

func TestParseCronAliases(t *testing.T) {
	for alias, want := range map[string]string{
		"@hourly": "0 * * * *",
		"@daily":  "0 0 * * *",
		"@weekly": "0 0 * * 0",
	} {
		t.Run(alias, func(t *testing.T) {
			schedule, err := parseCron(alias)
			require.NoError(t, err)
			expected, err := parseCron(want)
			require.NoError(t, err)
			assert.Equal(t, expected.minute.values, schedule.minute.values)
			assert.Equal(t, expected.hour.values, schedule.hour.values)
			assert.Equal(t, expected.weekday.values, schedule.weekday.values)
		})
	}
}

func TestNextTimesDailyAtNine(t *testing.T) {
	schedule, err := parseCron("0 9 * * *")
	require.NoError(t, err)
	from := time.Date(2026, 9, 17, 10, 30, 0, 0, time.UTC)
	times := schedule.nextTimes(from, 3)
	require.Len(t, times, 3)
	assert.Equal(t, time.Date(2026, 9, 18, 9, 0, 0, 0, time.UTC), times[0])
	assert.Equal(t, time.Date(2026, 9, 19, 9, 0, 0, 0, time.UTC), times[1])
	assert.Equal(t, time.Date(2026, 9, 20, 9, 0, 0, 0, time.UTC), times[2])
}

func TestNextTimesStepAndWeekday(t *testing.T) {
	// 每 15 分钟一次
	schedule, err := parseCron("*/15 * * * *")
	require.NoError(t, err)
	from := time.Date(2026, 9, 17, 10, 7, 0, 0, time.UTC)
	times := schedule.nextTimes(from, 3)
	require.Len(t, times, 3)
	assert.Equal(t, time.Date(2026, 9, 17, 10, 15, 0, 0, time.UTC), times[0])
	assert.Equal(t, time.Date(2026, 9, 17, 10, 30, 0, 0, time.UTC), times[1])

	// 仅周一 08:00（2026-09-17 为周四）
	monday, err := parseCron("0 8 * * 1")
	require.NoError(t, err)
	next := monday.nextTimes(from, 1)
	require.Len(t, next, 1)
	assert.Equal(t, time.Monday, next[0].Weekday())
	assert.Equal(t, 8, next[0].Hour())
}

func TestComputeNextTriggerUsesTaskTimezone(t *testing.T) {
	// 归一化是调用方（创建/更新/预览）的职责，落库与 computeNextTrigger 一律接收标准 5 位 Cron
	normalized, err := normalizeCron("daily@09:00")
	require.NoError(t, err)
	next, err := computeNextTrigger(normalized, "Asia/Shanghai")
	require.NoError(t, err)
	location, err := time.LoadLocation("Asia/Shanghai")
	require.NoError(t, err)
	assert.Equal(t, 9, next.In(location).Hour())
	assert.Equal(t, 0, next.In(location).Minute())
	assert.True(t, next.After(time.Now()))

	_, err = computeNextTrigger("0 9 * * *", "Not/AZone")
	require.Error(t, err)
}

func TestDescribeCron(t *testing.T) {
	cases := map[string]string{
		"* * * * *":  "每分钟",
		"0 * * * *":  "每小时整点",
		"30 * * * *": "每小时 30分",
		"0 9 * * *":  "每天 09点00分",
		"30 9 * * 1": "每周周一 09点30分",
		"0 9 15 * *": "每月15号 09点00分",
		"0 9 15 3 *": "每年3月15号 09点00分",
	}
	for cron, want := range cases {
		t.Run(cron, func(t *testing.T) {
			assert.Equal(t, want, describeCron(cron))
		})
	}
}
