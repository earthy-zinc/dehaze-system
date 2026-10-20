package handlers

import (
	"strconv"
	"strings"
)

func getAsString(row map[string]interface{}, key string) string {
	v, ok := row[key]
	if !ok || v == nil {
		return ""
	}
	return strings.TrimSpace(toString(v))
}

func toString(v interface{}) string {
	switch s := v.(type) {
	case string:
		return s
	case int:
		return strconv.Itoa(s)
	case int8:
		return strconv.Itoa(int(s))
	case int16:
		return strconv.Itoa(int(s))
	case int32:
		return strconv.Itoa(int(s))
	case int64:
		return strconv.FormatInt(s, 10)
	case uint:
		return strconv.FormatUint(uint64(s), 10)
	case uint8:
		return strconv.FormatUint(uint64(s), 10)
	case uint16:
		return strconv.FormatUint(uint64(s), 10)
	case uint32:
		return strconv.FormatUint(uint64(s), 10)
	case uint64:
		return strconv.FormatUint(s, 10)
	case float32:
		return strconv.FormatFloat(float64(s), 'f', -1, 32)
	case float64:
		return strconv.FormatFloat(s, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(s)
	default:
		return ""
	}
}

func parseInteger(row map[string]interface{}, key string, defaultValue int) int {
	v, ok := row[key]
	if !ok || v == nil {
		return defaultValue
	}
	switch n := v.(type) {
	case int:
		return n
	case int8:
		return int(n)
	case int16:
		return int(n)
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float32:
		return int(n)
	case float64:
		return int(n)
	case string:
		s := strings.TrimSpace(n)
		if s == "" {
			return defaultValue
		}
		i, err := strconv.Atoi(s)
		if err != nil {
			return defaultValue
		}
		return i
	default:
		return defaultValue
	}
}

func parseLong(row map[string]interface{}, key string, defaultValue int64) int64 {
	v, ok := row[key]
	if !ok || v == nil {
		return defaultValue
	}
	switch n := v.(type) {
	case int:
		return int64(n)
	case int8:
		return int64(n)
	case int16:
		return int64(n)
	case int32:
		return int64(n)
	case int64:
		return n
	case float32:
		return int64(n)
	case float64:
		return int64(n)
	case string:
		s := strings.TrimSpace(n)
		if s == "" {
			return defaultValue
		}
		i, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return defaultValue
		}
		return i
	default:
		return defaultValue
	}
}

func parseStatus(row map[string]interface{}, key string, defaultValue int) int {
	s := getAsString(row, key)
	if s == "" {
		return defaultValue
	}
	if s == "启用" {
		return 1
	}
	if s == "禁用" {
		return 0
	}
	return defaultValue
}

func parseBoolInt(row map[string]interface{}, key string, defaultValue int) int {
	s := getAsString(row, key)
	if s == "" {
		return defaultValue
	}
	if s == "是" {
		return 1
	}
	if s == "否" {
		return 0
	}
	return defaultValue
}
