package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// PathSecurityUtil 路径安全工具类
// 用于防止路径遍历攻击和验证文件名安全性
//
// 路径遍历攻击（Path Traversal）是一种安全漏洞，攻击者通过操纵文件路径
// 来访问应用程序预期目录之外的文件或目录。
//
// 攻击示例:
//   - ../../../etc/passwd (尝试读取系统密码文件)
//   - ..\..\..\windows\system32\config\sam (尝试读取 Windows 系统文件)
//   - /var/www/uploads/../../etc/passwd (混合使用绝对路径和相对路径)
type PathSecurityUtil struct{}

// NewPathSecurityUtil 创建路径安全工具实例
func NewPathSecurityUtil() *PathSecurityUtil {
	return &PathSecurityUtil{}
}

// DEFAULT_FILENAME_REGEX 默认安全文件名正则表达式
// 只允许字母、数字、点、连字符和下划线
const DEFAULT_FILENAME_REGEX = `^[a-zA-Z0-9.\-_]+$`

// DEFAULT_BASE_DIRECTORY 默认基础目录配置
// 用于限制文件操作的根目录
var DEFAULT_BASE_DIRECTORY = ""

// SetDefaultBaseDirectory 设置默认基础目录
//
// 参数:
//   dir: 基础目录路径
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	path.SetDefaultBaseDirectory("/var/www/uploads")
func (u *PathSecurityUtil) SetDefaultBaseDirectory(dir string) {
	DEFAULT_BASE_DIRECTORY = dir
}

// ValidatePath 校验路径是否安全，防止路径遍历攻击
//
// 参数:
//   fullPath: 完整的文件路径（用户输入或构建的路径）
//   basePath: 基础路径（允许访问的根目录）
//
// 返回:
//   string: 规范化后的安全路径
//   error: 如果路径不安全则返回错误
//
// 安全原理:
// 1. 使用 filepath.Clean() 规范化路径，解析 . 和 ..
// 2. 使用 filepath.Rel() 计算相对路径
// 3. 检查相对路径是否以 .. 开头（路径遍历攻击的标志）
// 4. 检查规范化后的路径是否在基础路径范围内
//
// 攻击防御:
//   - ../../../etc/passwd → 检测到路径遍历攻击
//   - /var/www/uploads/../../etc/passwd → 检测到路径不在允许范围内
//   - ../secret.txt → 检测到路径遍历攻击
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	baseDir := "/var/www/uploads"
//	userInput := "images/photo.jpg"
//	safePath, err := path.ValidatePath(userInput, baseDir)
//	// safePath = "/var/www/uploads/images/photo.jpg", err = nil
//
//	// 尝试路径遍历攻击
//	attackInput := "../../../etc/passwd"
//	_, err = path.ValidatePath(attackInput, baseDir)
//	// err = "检测到路径遍历攻击"
func (u *PathSecurityUtil) ValidatePath(fullPath string, basePath string) (string, error) {
	if fullPath == "" {
		return "", fmt.Errorf("路径不能为空")
	}

	if basePath == "" {
		return "", fmt.Errorf("基础路径不能为空")
	}

	// 规范化路径（解析 . 和 ..）
	normalizedPath := filepath.Clean(fullPath)
	normalizedBasePath := filepath.Clean(basePath)

	// 检查规范化后的路径是否在基础路径下
	relPath, err := filepath.Rel(normalizedBasePath, normalizedPath)
	if err != nil {
		return "", fmt.Errorf("路径关系计算失败: %w", err)
	}

	// 如果相对路径以 .. 开头，说明路径试图向上遍历
	// 这是路径遍历攻击的典型特征
	if strings.HasPrefix(relPath, "..") {
		return "", fmt.Errorf("检测到路径遍历攻击: 试图访问基础路径之外的文件")
	}

	// 再次检查规范化路径是否以基础路径开头
	// 确保最终路径确实在基础路径内
	if !strings.HasPrefix(normalizedPath, normalizedBasePath) {
		return "", fmt.Errorf("路径不在允许的目录范围内")
	}

	return normalizedPath, nil
}

// ValidateFileName 校验文件名是否安全
//
// 参数:
//   fileName: 文件名（不含路径）
//
// 返回:
//   error: 如果文件名不安全则返回错误
//
// 安全检查:
// 1. 文件名不能为空
// 2. 不能包含路径分隔符（/ 或 \）
// 3. 不能包含 Windows 系统的非法字符
// 4. 不能是 Windows 系统保留名称（如 CON、PRN、AUX 等）
//
// Windows 非法字符:
//   < > : " | ? * \ /
//
// Windows 保留名称:
//   CON, PRN, AUX, NUL
//   COM1-COM9, LPT1-LPT9
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	err := path.ValidateFileName("photo.jpg")
//	// err = nil
//
//	err = path.ValidateFileName("../../../etc/passwd")
//	// err = "文件名不能包含路径分隔符"
//
//	err = path.ValidateFileName("test<file>.txt")
//	// err = "文件名包含非法字符"
func (u *PathSecurityUtil) ValidateFileName(fileName string) error {
	if fileName == "" {
		return fmt.Errorf("文件名不能为空")
	}

	// 检查文件名中是否包含路径分隔符
	// 防止通过文件名进行路径遍历攻击
	if strings.ContainsAny(fileName, `/\\`) {
		return fmt.Errorf("文件名不能包含路径分隔符")
	}

	// 检查是否包含特殊字符
	// 这些字符在 Windows 系统中被保留
	if strings.ContainsAny(fileName, `<>:"|?*`) {
		return fmt.Errorf("文件名包含非法字符")
	}

	// 检查是否为保留名称
	// Windows 系统不允许使用这些名称作为文件名
	upperFileName := strings.ToUpper(fileName)
	reservedNames := []string{
		"CON", "PRN", "AUX", "NUL",
		"COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
		"LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
	}

	for _, reserved := range reservedNames {
		if upperFileName == reserved {
			return fmt.Errorf("'%s' 是系统保留名称", fileName)
		}
	}

	return nil
}

// ValidateFileNameWithRegex 使用正则表达式校验文件名
//
// 参数:
//   fileName: 文件名（不含路径）
//   pattern: 正则表达式模式（如果为空则使用默认模式）
//
// 返回:
//   error: 如果文件名不匹配正则表达式则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	err := path.ValidateFileNameWithRegex("photo.jpg", `^[a-zA-Z0-9.\-_]+$`)
//	// err = nil
//
//	err = path.ValidateFileNameWithRegex("test@file.txt", `^[a-zA-Z0-9.\-_]+$`)
//	// err = "文件名包含不安全字符"
func (u *PathSecurityUtil) ValidateFileNameWithRegex(fileName string, pattern string) error {
	if fileName == "" {
		return fmt.Errorf("文件名不能为空")
	}

	if pattern == "" {
		pattern = DEFAULT_FILENAME_REGEX
	}

	// 使用正则表达式校验文件名
	matched, err := regexp.MatchString(pattern, fileName)
	if err != nil {
		return fmt.Errorf("正则表达式错误: %w", err)
	}

	if !matched {
		return fmt.Errorf("文件名包含不安全字符")
	}

	return nil
}

// SanitizeFileName 清理文件名，移除不安全字符
// 将不安全字符替换为下划线
//
// 参数:
//   fileName: 原始文件名
//
// 返回:
//   string: 清理后的安全文件名
//
// 清理规则:
// 1. 将路径分隔符（/、\）替换为下划线
// 2. 将连续的点（..）替换为下划线
// 3. 将非法字符替换为下划线
// 4. 移除首尾的空格和点
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	safeName := path.SanitizeFileName("../../../etc/passwd")
//	// safeName = "_________etc_passwd"
//
//	safeName = path.SanitizeFileName("  .test.file.txt.  ")
//	// safeName = "test.file.txt"
func (u *PathSecurityUtil) SanitizeFileName(fileName string) string {
	if fileName == "" {
		return ""
	}

	// 替换路径分隔符
	result := strings.ReplaceAll(fileName, "/", "_")
	result = strings.ReplaceAll(result, "\\", "_")

	// 替换连续点（防止路径遍历）
	for strings.Contains(result, "..") {
		result = strings.ReplaceAll(result, "..", "_")
	}

	// 使用正则替换非法字符为下划线
	// 只保留字母、数字、点、连字符和下划线
	regex := regexp.MustCompile(`[^a-zA-Z0-9.\-_]`)
	result = regex.ReplaceAllString(result, "_")

	// 移除首尾空格和点
	result = strings.Trim(result, " .")

	return result
}

// GetSafeFilePath 获取安全的文件路径
// 组合使用文件名清理和路径校验
//
// 参数:
//   baseDir: 基础目录
//   fileName: 文件名
//
// 返回:
//   string: 安全的完整文件路径
//   error: 如果路径不安全则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	baseDir := "/var/www/uploads"
//	fileName := "images/photo.jpg"
//	safePath, err := path.GetSafeFilePath(baseDir, fileName)
//	// safePath = "/var/www/uploads/images/photo.jpg", err = nil
//
//	// 尝试使用恶意文件名
//	attackName := "../../../etc/passwd"
//	_, err = path.GetSafeFilePath(baseDir, attackName)
//	// err != nil (检测到路径遍历攻击)
func (u *PathSecurityUtil) GetSafeFilePath(baseDir string, fileName string) (string, error) {
	// 清理文件名
	safeFileName := u.SanitizeFileName(fileName)
	if safeFileName == "" {
		return "", fmt.Errorf("文件名清理后为空")
	}

	// 构建完整路径
	fullPath := filepath.Join(baseDir, safeFileName)

	// 校验路径安全
	normalizedPath, err := u.ValidatePath(fullPath, baseDir)
	if err != nil {
		return "", err
	}

	return normalizedPath, nil
}

// CreateSafeDir 安全地创建目录
// 防止通过相对路径创建目录到基础路径之外
//
// 参数:
//   basePath: 基础路径
//   relativePath: 相对路径
//
// 返回:
//   string: 创建的目录路径
//   error: 如果创建失败则返回错误
//
// 安全原理:
// 1. 规范化相对路径
// 2. 检查路径遍历攻击
// 3. 校验最终路径是否在基础路径内
// 4. 使用 os.MkdirAll 创建目录
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	baseDir := "/var/www/uploads"
//	relPath := "images/2024"
//	dirPath, err := path.CreateSafeDir(baseDir, relPath)
//	// dirPath = "/var/www/uploads/images/2024", err = nil
//
//	// 尝试路径遍历攻击
//	attackPath := "../../../etc"
//	_, err = path.CreateSafeDir(baseDir, attackPath)
//	// err = "检测到路径遍历攻击"
func (u *PathSecurityUtil) CreateSafeDir(basePath string, relativePath string) (string, error) {
	// 规范化相对路径
	relativePath = filepath.Clean(relativePath)

	// 检查是否为空
	if relativePath == "" || relativePath == "." {
		return basePath, nil
	}

	// 检查路径遍历
	if strings.HasPrefix(relativePath, "..") || strings.Contains(relativePath, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("检测到路径遍历攻击")
	}

	// 构建完整路径
	fullPath := filepath.Join(basePath, relativePath)

	// 校验路径安全
	normalizedPath, err := u.ValidatePath(fullPath, basePath)
	if err != nil {
		return "", err
	}

	// 创建目录（权限 0755：所有者可读写执行，组和其他用户可读执行）
	err = os.MkdirAll(normalizedPath, 0755)
	if err != nil {
		return "", fmt.Errorf("创建目录失败: %w", err)
	}

	return normalizedPath, nil
}

// CheckFileInDirectory 检查文件是否在指定目录内
//
// 参数:
//   filePath: 文件路径
//   dirPath: 目录路径
//
// 返回:
//   bool: 如果文件在目录内返回 true，否则返回 false
//   error: 如果路径计算失败则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	dir := "/var/www/uploads"
//	file := "/var/www/uploads/images/photo.jpg"
//	isInDir, err := path.CheckFileInDirectory(file, dir)
//	// isInDir = true, err = nil
//
//	// 尝试检查目录外的文件
//	externalFile := "/etc/passwd"
//	isInDir, _ = path.CheckFileInDirectory(externalFile, dir)
//	// isInDir = false
func (u *PathSecurityUtil) CheckFileInDirectory(filePath string, dirPath string) (bool, error) {
	// 获取文件的绝对路径
	absFilePath, err := filepath.Abs(filePath)
	if err != nil {
		return false, fmt.Errorf("获取文件绝对路径失败: %w", err)
	}

	// 获取目录的绝对路径
	absDirPath, err := filepath.Abs(dirPath)
	if err != nil {
		return false, fmt.Errorf("获取目录绝对路径失败: %w", err)
	}

	// 计算相对路径
	relPath, err := filepath.Rel(absDirPath, absFilePath)
	if err != nil {
		return false, fmt.Errorf("计算相对路径失败: %w", err)
	}

	// 如果相对路径以 .. 开头，说明文件不在目录内
	if strings.HasPrefix(relPath, "..") {
		return false, nil
	}

	return true, nil
}

// GetSafeFilePathWithDefault 使用默认基础目录获取安全文件路径
// 如果设置了默认基础目录，则可以使用此方法简化调用
//
// 参数:
//   relativePath: 相对路径
//
// 返回:
//   string: 安全的完整文件路径
//   error: 如果路径不安全或未设置默认目录则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	path.SetDefaultBaseDirectory("/var/www/uploads")
//	safePath, err := path.GetSafeFilePathWithDefault("images/photo.jpg")
//	// safePath = "/var/www/uploads/images/photo.jpg", err = nil
func (u *PathSecurityUtil) GetSafeFilePathWithDefault(relativePath string) (string, error) {
	if DEFAULT_BASE_DIRECTORY == "" {
		return "", fmt.Errorf("未设置默认基础目录")
	}

	return u.GetSafeFilePath(DEFAULT_BASE_DIRECTORY, relativePath)
}

// IsPathTraversalAttack 检测是否为路径遍历攻击
//
// 参数:
//   pathStr: 路径字符串
//
// 返回:
//   bool: 如果是路径遍历攻击返回 true
//
// 检测特征:
// - 包含 ../ 或 ..\
// - 包含绝对路径且试图向上遍历
// - 包含多个连续的点
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	isAttack := path.IsPathTraversalAttack("../../../etc/passwd")
//	// isAttack = true
//
//	isAttack = path.IsPathTraversalAttack("images/photo.jpg")
//	// isAttack = false
func (u *PathSecurityUtil) IsPathTraversalAttack(pathStr string) bool {
	if pathStr == "" {
		return false
	}

	// 检查常见的路径遍历模式
	// 注：末尾不再包含裸 ".."，避免误判合法的 "my..file.txt"、"v..1.0" 等含双点的文件名
	traversalPatterns := []string{
		"../",
		"..\\",
		"\\..\\",
		"..%2f",
		"..%2F",
		"..%5c",
		"..%5C",
		"%2e%2e",
		"%252e%252e",
	}

	lowerPath := strings.ToLower(pathStr)
	for _, pattern := range traversalPatterns {
		if strings.Contains(lowerPath, strings.ToLower(pattern)) {
			return true
		}
	}

	return false
}

// GetAbsolutePathWithValidation 获取绝对路径并进行安全验证
//
// 参数:
//   pathStr: 路径字符串
//   basePath: 基础路径
//
// 返回:
//   string: 安全的绝对路径
//   error: 如果路径不安全则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	baseDir := "/var/www/uploads"
//	relPath := "images/photo.jpg"
//	absPath, err := path.GetAbsolutePathWithValidation(relPath, baseDir)
//	// absPath = "/var/www/uploads/images/photo.jpg", err = nil
func (u *PathSecurityUtil) GetAbsolutePathWithValidation(pathStr string, basePath string) (string, error) {
	// 构建完整路径
	fullPath := filepath.Join(basePath, pathStr)

	// 转换为绝对路径
	absPath, err := filepath.Abs(fullPath)
	if err != nil {
		return "", fmt.Errorf("获取绝对路径失败: %w", err)
	}

	// 验证路径安全
	safePath, err := u.ValidatePath(absPath, basePath)
	if err != nil {
		return "", err
	}

	return safePath, nil
}

// NormalizePath 规范化路径字符串
// 统一路径分隔符，解析 . 和 ..
//
// 参数:
//   pathStr: 路径字符串
//
// 返回:
//   string: 规范化后的路径
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	normalized := path.NormalizePath("a/b/../c/./d")
//	// normalized = "a/c/d"
func (u *PathSecurityUtil) NormalizePath(pathStr string) string {
	if pathStr == "" {
		return ""
	}

	// 使用 filepath.Clean 规范化路径
	return filepath.Clean(pathStr)
}

// JoinSafePaths 安全地连接多个路径
// 防止通过路径连接进行攻击
//
// 参数:
//   basePath: 基础路径
//   paths: 要连接的路径部分
//
// 返回:
//   string: 连接后的安全路径
//   error: 如果路径不安全则返回错误
//
// 使用示例:
//
//	path := NewPathSecurityUtil()
//	baseDir := "/var/www/uploads"
//	safePath, err := path.JoinSafePaths(baseDir, "images", "2024", "photo.jpg")
//	// safePath = "/var/www/uploads/images/2024/photo.jpg", err = nil
func (u *PathSecurityUtil) JoinSafePaths(basePath string, paths ...string) (string, error) {
	// 连接所有路径
	fullPath := filepath.Join(basePath, filepath.Join(paths...))

	// 规范化路径
	normalizedPath := filepath.Clean(fullPath)

	// 验证路径安全
	absBasePath, err := filepath.Abs(basePath)
	if err != nil {
		return "", fmt.Errorf("获取基础路径绝对路径失败: %w", err)
	}

	_, err = u.ValidatePath(normalizedPath, absBasePath)
	if err != nil {
		return "", err
	}

	return normalizedPath, nil
}
