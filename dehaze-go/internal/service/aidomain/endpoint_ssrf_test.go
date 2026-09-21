package aidomain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestIsSafeURLRejectsInternalTargets SSRF 回归护栏：只有公网 https 放行。
//
// 覆盖曾真实放行的三类绕过：IPv6 链路本地 fe80::/10、IPv4 映射 ::ffff:127.0.0.1、
// 字符串前缀判定漏掉的 IPv4 段（CGNAT / 组播 / 保留）。
func TestIsSafeURLRejectsInternalTargets(t *testing.T) {
	denied := []string{
		"",
		"   ",
		"http://example.com/",
		"ftp://127.0.0.1/",
		"https://localhost/",
		"https://agent.local/",
		"https://agent.internal/",
		// IPv4 回环/私有/链路本地/未指定
		"https://127.0.0.1/",
		"https://127.8.8.8/",
		"https://10.0.0.1/",
		"https://172.16.0.1/",
		"https://172.31.255.255/",
		"https://192.168.1.1/",
		"https://169.254.169.254/",
		"https://0.0.0.0/",
		"https://0.1.2.3/",
		// 安全策略额外封禁段与 CGNAT / 组播 / 保留
		"https://9.9.9.9/",
		"https://11.0.0.1/",
		"https://21.0.0.1/",
		"https://30.0.0.1/",
		"https://100.64.0.1/",
		"https://100.127.255.255/",
		"https://224.0.0.1/",
		"https://240.0.0.1/",
		// 文档段（TEST-NET-1/2/3）与基准测试段 198.18.0.0/15：python ipaddress.is_private 覆盖
		"https://192.0.2.1/",
		"https://192.0.2.255/",
		"https://198.51.100.1/",
		"https://203.0.113.9/",
		"https://198.18.0.1/",
		"https://198.19.255.255/",
		// IPv6：回环 / 未指定 / 链路本地 / ULA
		"https://[::1]/",
		"https://[::]/",
		"https://[fe80::1]/",
		"https://[fd00::1]/",
		"https://[fc00::1]/",
		"https://[ff02::1]/",
		// IPv4 映射形式（按映射到的 IPv4 判定）
		"https://[::ffff:127.0.0.1]/",
		"https://[::ffff:10.0.0.1]/",
		"https://[::ffff:192.168.1.1]/",
		// 域名解析失败保守拒绝
		"https://no-such-host-for-ssrf-test.invalid/",
	}
	for _, raw := range denied {
		t.Run("deny "+raw, func(t *testing.T) {
			assert.False(t, isSafeURL(raw), "不应放行: %s", raw)
		})
	}
}

// TestIsSafeURLAllowsPublicTargets 避免过度封禁：公网字面 IP 必须放行（域名走 DNS，测试环境不保证网络）。
func TestIsSafeURLAllowsPublicTargets(t *testing.T) {
	allowed := []string{
		"https://8.8.8.8/",
		"https://1.2.3.4/",
		"https://172.32.0.1/",
		"https://172.15.255.255/",
		"https://100.63.255.255/",
		"https://100.128.0.1/",
		"https://223.255.255.255/",
		// 保留/文档段必须是精确前缀：同 /16 但不在 TEST-NET-2、198.20/15 之外仍放行
		"https://198.51.99.1/",
		"https://198.20.0.1/",
		"https://192.0.3.1/",
		"https://203.0.114.1/",
		"https://192.88.99.1/",
		"https://[2001:4860:4860::8888]/",
		"https://[::ffff:8.8.8.8]/",
	}
	for _, raw := range allowed {
		t.Run("allow "+raw, func(t *testing.T) {
			assert.True(t, isSafeURL(raw), "应放行: %s", raw)
		})
	}
}
