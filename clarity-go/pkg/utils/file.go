package utils

import "os"

// DeleteTempFile 删除临时文件
func DeleteTempFile(filePath string) error {
	if filePath == "" {
		return nil
	}
	return os.Remove(filePath)
}
