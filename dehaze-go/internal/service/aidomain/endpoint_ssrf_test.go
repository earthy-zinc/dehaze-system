package aidomain

import (
	"errors"
	"net"
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
	}
	for _, raw := range denied {
		t.Run("deny "+raw, func(t *testing.T) {
			assert.False(t, isSafeURL(raw), "不应放行: %s", raw)
		})
	}
}

// TestIsSafeURLDomainResolution 域名解析分支：替换解析入口注入结果，不依赖真实 DNS
// （内网 DNS 会把不存在的域名通配解析到某个地址，按真实解析结果断言会随环境假绿/假红）。
func TestIsSafeURLDomainResolution(t *testing.T) {
	original := lookupIP
	t.Cleanup(func() { lookupIP = original })

	cases := []struct {
		name      string
		addresses []net.IP
		err       error
		want      bool
	}{
		{"解析失败保守拒绝", nil, errors.New("no such host"), false},
		{"解析结果为空保守拒绝", []net.IP{}, nil, false},
		{"解析到内网拒绝", []net.IP{net.ParseIP("192.168.50.10")}, nil, false},
		{"任一地址为内网即拒绝", []net.IP{net.ParseIP("93.184.216.34"), net.ParseIP("10.1.2.3")}, nil, false},
		{"解析到公网放行", []net.IP{net.ParseIP("93.184.216.34")}, nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lookupIP = func(string) ([]net.IP, error) { return tc.addresses, tc.err }
			assert.Equal(t, tc.want, isSafeURL("https://agent.example.com/x"))
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
