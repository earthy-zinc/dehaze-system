package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// GoRoot 返回 dehaze-go 服务根目录：以 go.mod 为标记从调用点向上查找，与进程工作目录解耦。
// 源码树不可见时（二进制被单独拷贝执行）退化为从当前工作目录向上查找，最后兜底为工作目录。
func GoRoot() string {
	if _, file, _, ok := runtime.Caller(0); ok {
		if dir := findUpward(filepath.Dir(file), "go.mod"); dir != "" {
			return dir
		}
	}

	wd, err := os.Getwd()
	if err != nil {
		panic(fmt.Sprintf("config: 获取工作目录失败: %v", err))
	}
	if dir := findUpward(wd, "go.mod"); dir != "" {
		return dir
	}
	return wd
}

// RepoRoot 返回仓库根目录：根 .env 与三端共享数据（data/upload、datasets）的基准。
func RepoRoot() string {
	return filepath.Dir(GoRoot())
}

func findUpward(dir, marker string) string {
	for {
		if _, err := os.Stat(filepath.Join(dir, marker)); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// normalizePaths 把配置里的相对路径转为锚定后的绝对路径：
// 服务自身运行产物（日志、SQLite）相对服务根，三端共享的上传目录相对仓库根；绝对路径原样保留。
// 配置文件里刻意留空（如 uploadPath: "${UPLOAD_DIR}" 未设置）时回落到 fallback 默认值。
func normalizePaths(cfg *AppConfig) {
	cfg.Zap.Directory = resolveAgainst(GoRoot(), cfg.Zap.Directory, "logs")
	if cfg.DB.SQLite != nil {
		cfg.DB.SQLite.Path = resolveAgainst(GoRoot(), cfg.DB.SQLite.Path, filepath.Join("data", "dehaze.db"))
	}
	cfg.File.Storage.Local.UploadPath = resolveAgainst(RepoRoot(), cfg.File.Storage.Local.UploadPath, filepath.Join("data", "upload"))
}

func resolveAgainst(base, path, fallback string) string {
	if path == "" {
		path = fallback
	}
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(base, path)
}
