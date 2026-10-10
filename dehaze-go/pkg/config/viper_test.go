package config

import (
	"os"
	"strings"
	"testing"
)

// expandEnv 的契约：${VAR} 必填（缺失即报错），${VAR:default} / ${VAR:} 缺失取默认值，
// 变量已设置但为空串时按显式空值处理。环境身份类变量（库 / bucket / vhost / 索引前缀）
// 依赖该契约在启动时暴露缺失，而不是静默展开为空写进生产预留命名空间。
func TestExpandEnv(t *testing.T) {
	const present = "DEHAZE_EXPAND_TEST_PRESENT"
	const absent = "DEHAZE_EXPAND_TEST_ABSENT"
	t.Setenv(present, "dev_")
	if err := os.Unsetenv(absent); err != nil {
		t.Fatalf("清理 %s 失败: %v", absent, err)
	}

	cases := []struct {
		name    string
		content string
		want    string
	}{
		{"已设置取真实值", "${" + present + "}", "dev_"},
		{"已设置时默认值不生效", "${" + present + ":fallback}", "dev_"},
		{"未设置取默认值", "${" + absent + ":fallback}", "fallback"},
		{"未设置取显式空默认", "${" + absent + ":}", ""},
		{"无占位符原样返回", "index-prefix: dev_", "index-prefix: dev_"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expandEnv(tc.content)
			if err != nil {
				t.Fatalf("expandEnv(%q) 意外报错: %v", tc.content, err)
			}
			if got != tc.want {
				t.Fatalf("expandEnv(%q) = %q, want %q", tc.content, got, tc.want)
			}
		})
	}
}

func TestExpandEnvMissingRequiredFails(t *testing.T) {
	const missingA = "DEHAZE_EXPAND_TEST_MISSING_A"
	const missingB = "DEHAZE_EXPAND_TEST_MISSING_B"
	for _, name := range []string{missingA, missingB} {
		if err := os.Unsetenv(name); err != nil {
			t.Fatalf("清理 %s 失败: %v", name, err)
		}
	}

	_, err := expandEnv("database: \"${" + missingA + "}\"\nprefix: \"${" + missingB + "}\"")
	if err == nil {
		t.Fatal("缺少无默认值的占位符变量时应报错")
	}
	// 一次报全且信息可定位（按名排序，避免顺序抖动）
	for _, name := range []string{missingA, missingB} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("错误信息 %q 未包含缺失变量 %s", err.Error(), name)
		}
	}
}

func TestExpandEnvExplicitEmptyValueWins(t *testing.T) {
	const name = "DEHAZE_EXPAND_TEST_EMPTY"
	t.Setenv(name, "")

	got, err := expandEnv("index-prefix: \"${" + name + ":fallback}\"")
	if err != nil {
		t.Fatalf("显式空值不应报错: %v", err)
	}
	if got != `index-prefix: ""` {
		t.Fatalf("显式空值应保留为空串，得到 %q", got)
	}
}
