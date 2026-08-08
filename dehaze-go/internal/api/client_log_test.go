package api

import (
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// newObservedLogger 创建带 observer 的 zap.Logger，便于断言日志输出
func newObservedLogger() (*zap.Logger, *observer.ObservedLogs) {
	core, recorded := observer.New(zapcore.InfoLevel)
	return zap.New(core), recorded
}

func TestIsError(t *testing.T) {
	if !isError("ERROR") || !isError("error") || !isError(" Error ") {
		t.Fatal("isError 应识别大小写不敏感及带空格的 ERROR")
	}
	if isError("WARN") || isError("INFO") || isError("") {
		t.Fatal("isError 不应误判非 ERROR 级别")
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("short", 2000); got != "short" {
		t.Fatalf("短字符串不应截断，got %q", got)
	}
	got := truncate(strings.Repeat("m", 3000), maxMessageLength)
	if len(got) != maxMessageLength {
		t.Fatalf("应截断到 %d，got %d", maxMessageLength, len(got))
	}
	// error_stack 截断到 8000
	longStack := strings.Repeat("s", 10000)
	if got := truncate(longStack, maxErrorStackLen); len(got) != maxErrorStackLen {
		t.Fatalf("error_stack 应截断到 %d，got %d", maxErrorStackLen, len(got))
	}
}

func TestWriteEntry_AnonymousErrorWithTraceIDWrites(t *testing.T) {
	log, recorded := newObservedLogger()
	entry := &dto.ClientLogEntry{
		Level:   "ERROR",
		TraceID: "trace-abc",
		Message: "test message",
		App:     "react",
	}

	writeEntry(entry, 0, log) // userID=0 表示匿名

	logs := recorded.All()
	if len(logs) != 1 {
		t.Fatalf("匿名 ERROR 带 trace_id 应落盘 1 条，got %d", len(logs))
	}
	if logs[0].Level != zapcore.ErrorLevel {
		t.Fatalf("级别应为 ERROR，got %v", logs[0].Level)
	}
	if logs[0].Message != "test message" {
		t.Fatalf("消息不匹配，got %q", logs[0].Message)
	}
	// trace_id 应作为字段注入
	if logs[0].ContextMap()["trace_id"] != "trace-abc" {
		t.Fatalf("trace_id 字段应注入，got %v", logs[0].ContextMap()["trace_id"])
	}
}

func TestWriteEntry_AnonymousWarnDropped(t *testing.T) {
	log, recorded := newObservedLogger()
	entry := &dto.ClientLogEntry{
		Level:   "WARN",
		TraceID: "trace-abc",
		Message: "should drop",
	}

	writeEntry(entry, 0, log)

	if len(recorded.All()) != 0 {
		t.Fatal("匿名 WARN 应被丢弃")
	}
}

func TestWriteEntry_AnonymousErrorWithoutTraceIDDropped(t *testing.T) {
	log, recorded := newObservedLogger()
	entry := &dto.ClientLogEntry{
		Level:   "ERROR",
		TraceID: "",
		Message: "should drop",
	}

	writeEntry(entry, 0, log)

	if len(recorded.All()) != 0 {
		t.Fatal("匿名 ERROR 无 trace_id 应被丢弃")
	}
}

func TestWriteEntry_LoggedInUserAllLevelsWrite(t *testing.T) {
	log, recorded := newObservedLogger()

	// 已登录用户（userID > 0）INFO/WARN/ERROR 均应落盘
	levels := []string{"INFO", "WARN", "ERROR"}
	for _, level := range levels {
		entry := &dto.ClientLogEntry{
			Level:   level,
			TraceID: "",
			Message: "logged in",
		}
		writeEntry(entry, 42, log)
	}

	logs := recorded.All()
	if len(logs) != 3 {
		t.Fatalf("已登录用户三级别应落盘 3 条，got %d", len(logs))
	}
	// user_id 字段应注入
	for _, l := range logs {
		if l.ContextMap()["user_id"] != int64(42) {
			t.Fatalf("user_id 字段应注入为 42，got %v", l.ContextMap()["user_id"])
		}
	}
}

func TestWriteEntry_LevelNormalization(t *testing.T) {
	log, recorded := newObservedLogger()

	// 级别大小写不敏感、空白/null 默认 INFO
	cases := []struct {
		input    string
		expected zapcore.Level
	}{
		{"error", zapcore.ErrorLevel},
		{"Warn", zapcore.WarnLevel},
		{"", zapcore.InfoLevel},
		{"   ", zapcore.InfoLevel},
	}
	for _, c := range cases {
		entry := &dto.ClientLogEntry{
			Level:   c.input,
			TraceID: "t",
			Message: "level test",
		}
		writeEntry(entry, 1, log)
	}

	logs := recorded.All()
	if len(logs) != len(cases) {
		t.Fatalf("应落盘 %d 条，got %d", len(cases), len(logs))
	}
	for i, c := range cases {
		if logs[i].Level != c.expected {
			t.Fatalf("case %d 级别应为 %v，got %v", i, c.expected, logs[i].Level)
		}
	}
}

func TestWriteEntry_MessageAndErrorStackTruncated(t *testing.T) {
	log, recorded := newObservedLogger()
	entry := &dto.ClientLogEntry{
		Level:      "ERROR",
		TraceID:    "trace-abc",
		Message:    strings.Repeat("m", 3000),
		ErrorStack: strings.Repeat("s", 10000),
	}

	writeEntry(entry, 0, log)

	logs := recorded.All()
	if len(logs) != 1 {
		t.Fatalf("应落盘 1 条，got %d", len(logs))
	}
	if len(logs[0].Message) != maxMessageLength {
		t.Fatalf("message 应截断到 %d，got %d", maxMessageLength, len(logs[0].Message))
	}
	// error_stack 字段应截断到 8000
	stack := logs[0].ContextMap()["error_stack"]
	if s, ok := stack.(string); !ok || len(s) != maxErrorStackLen {
		t.Fatalf("error_stack 应截断到 %d，got %v", maxErrorStackLen, stack)
	}
}

func TestBuildFields_BlankStringExcluded(t *testing.T) {
	entry := &dto.ClientLogEntry{
		App:        "react",
		URL:        "   ",   // 纯空白：不应进入（与 Java isNotBlank 对齐）
		UserAgent:  "",      // 空白：不应进入
		Method:     "POST",
		Path:       "",      // 空白：不应进入
		ErrorStack: "stack", // 非空：应进入
	}

	log, recorded := newObservedLogger()
	log.Info("test", buildFields(entry, "trace-abc")...)

	ctx := recorded.All()[0].ContextMap()
	if ctx["app"] != "react" {
		t.Error("app 字段应进入")
	}
	if ctx["method"] != "POST" {
		t.Error("method 字段应进入")
	}
	if ctx["error_stack"] != "stack" {
		t.Error("error_stack 字段应进入")
	}
	if _, ok := ctx["url"]; ok {
		t.Error("url 纯空白不应进入")
	}
	if _, ok := ctx["user_agent"]; ok {
		t.Error("user_agent 空白不应进入")
	}
	if _, ok := ctx["path"]; ok {
		t.Error("path 空白不应进入")
	}
}

func TestBuildFields_NumericNullable(t *testing.T) {
	status := 500
	duration := 1203.5
	entry := &dto.ClientLogEntry{
		TraceID:     "t1",
		Status:      &status,
		Duration:    &duration,
		MetricValue: nil, // null：不应进入
	}

	log, recorded := newObservedLogger()
	log.Info("test", buildFields(entry, "t1")...)

	ctx := recorded.All()[0].ContextMap()
	// zap.Int 内部存储为 int64
	if ctx["status"] != int64(status) {
		t.Errorf("status 应注入为 %d，got %v", status, ctx["status"])
	}
	if ctx["duration"] != duration {
		t.Errorf("duration 应注入为 %v，got %v", duration, ctx["duration"])
	}
	if _, ok := ctx["metric_value"]; ok {
		t.Error("metric_value 为 null 不应进入")
	}
}
