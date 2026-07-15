package utils

import (
	"crypto/md5"
	"encoding/hex"
)

// MD5Hex 计算字符串的 MD5 十六进制表示（32 位小写）
func MD5Hex(s string) string {
	h := md5.Sum([]byte(s))
	return hex.EncodeToString(h[:])
}
