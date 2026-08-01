package logger

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Cutter struct {
	level        string
	layout       string
	formats      []string
	director     string
	retentionDay int
	maxSize      int64 // 单文件大小上限（字节），0 表示不按大小切割
	file         *os.File
	currentDate  string
	mutex        sync.Mutex
}

type CutterOption func(*Cutter)

// CutterWithLayout 时间格式
func CutterWithLayout(layout string) CutterOption {
	return func(c *Cutter) {
		c.layout = layout
	}
}

// CutterWithFormats 格式化参数
func CutterWithFormats(format ...string) CutterOption {
	return func(c *Cutter) {
		if len(format) > 0 {
			c.formats = format
		}
	}
}

// CutterWithMaxSize 设置单文件大小上限（字节），超限归档为 {level}.{n}.log 并开新活动文件
func CutterWithMaxSize(maxSize int64) CutterOption {
	return func(c *Cutter) {
		c.maxSize = maxSize
	}
}

func NewCutter(director string, level string, retentionDay int, options ...CutterOption) *Cutter {
	rotate := &Cutter{
		level:        level,
		director:     director,
		retentionDay: retentionDay,
	}
	for i := 0; i < len(options); i++ {
		options[i](rotate)
	}
	return rotate
}

func (c *Cutter) Write(bytes []byte) (n int, err error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	currentDate := time.Now().Format(time.DateOnly)
	if c.file == nil || c.currentDate != currentDate {
		if err := c.rotateByDate(currentDate); err != nil {
			return 0, err
		}
	}

	n, err = c.file.Write(bytes)
	if err != nil {
		return n, err
	}

	// 写入后检查大小，超限则归档当前文件并开新活动文件
	if c.maxSize > 0 {
		if info, statErr := c.file.Stat(); statErr == nil && info.Size() >= c.maxSize {
			if archiveErr := c.archiveCurrent(); archiveErr != nil {
				log.Printf("[WARN] 日志按大小归档失败: %v\n", archiveErr)
			}
		}
	}
	return n, nil
}

// filePath 返回指定日期的活动日志文件路径（logs/{date}/{level}.log 或含 formats 子段）
func (c *Cutter) filePath(date string) string {
	values := make([]string, 0, 3+len(c.formats))
	values = append(values, c.director)
	if c.layout != "" {
		values = append(values, date)
	}
	values = append(values, c.formats...)
	values = append(values, c.level+".log")
	return filepath.Join(values...)
}

// rotateByDate 跨天切换到新日期目录
func (c *Cutter) rotateByDate(currentDate string) error {
	if c.file != nil {
		if err := c.file.Close(); err != nil {
			return fmt.Errorf("关闭旧日志文件失败: %w", err)
		}
	}

	c.currentDate = currentDate
	path := c.filePath(currentDate)
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("创建日志目录失败: %w", err)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("打开日志文件失败: %w", err)
	}

	c.file = file
	c.currentDate = currentDate

	go c.cleanOldLogs()

	return nil
}

// archiveCurrent 将当前活动文件归档为 {level}.{n}.log 并开新活动文件
func (c *Cutter) archiveCurrent() error {
	if c.file != nil {
		_ = c.file.Close()
	}

	currentPath := c.filePath(c.currentDate)
	archivedPath := nextArchivedPath(filepath.Dir(currentPath), strings.TrimSuffix(filepath.Base(currentPath), ".log"))
	if err := os.Rename(currentPath, archivedPath); err != nil && !os.IsNotExist(err) {
		// 归档失败则重新打开原文件继续追加，避免丢日志
		if file, openErr := os.OpenFile(currentPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); openErr == nil {
			c.file = file
		}
		return err
	}

	file, err := os.OpenFile(currentPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("打开新日志文件失败: %w", err)
	}
	c.file = file
	return nil
}

// nextArchivedPath 在 dir 下找 {stem}.{n}.log 的最大序号 +1，返回归档路径
func nextArchivedPath(dir, stem string) string {
	n := 0
	prefix := stem + "."
	if entries, err := os.ReadDir(dir); err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".log") {
				continue
			}
			num := strings.TrimSuffix(strings.TrimPrefix(name, prefix), ".log")
			if i, atoiErr := strconv.Atoi(num); atoiErr == nil && i > n {
				n = i
			}
		}
	}
	return filepath.Join(dir, fmt.Sprintf("%s.%d.log", stem, n+1))
}

// archiveStartupLogs 启动时归档当天已存在的活动日志文件（dev 用，prod 不调用）
func archiveStartupLogs(directory string, levels []string) {
	today := time.Now().Format(time.DateOnly)
	dir := filepath.Join(directory, today)
	for _, level := range levels {
		path := filepath.Join(dir, level+".log")
		info, err := os.Stat(path)
		if err != nil || info.Size() == 0 {
			continue
		}
		archived := nextArchivedPath(dir, level)
		if err := os.Rename(path, archived); err != nil {
			log.Printf("[WARN] 启动归档 %s 失败: %v", path, err)
		}
	}
}

func (c *Cutter) cleanOldLogs() {
	if err := removeNDaysFolders(c.director, c.retentionDay); err != nil {
		// 使用标准 log 输出（避免循环依赖 pkg/logger → Cutter → pkg/logger）
		log.Printf("[WARN] 清理过期日志失败: %v\n", err)
	}
}

func (c *Cutter) Sync() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.file != nil {
		return c.file.Sync()
	}
	return nil
}

func (c *Cutter) Close() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if c.file != nil {
		err := c.file.Close()
		c.file = nil
		return err
	}
	return nil
}

// 增加日志目录文件清理 小于等于零的值默认忽略不再处理
func removeNDaysFolders(dir string, days int) error {
	if days <= 0 {
		return nil
	}
	cutoff := time.Now().AddDate(0, 0, -days)
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && info.ModTime().Before(cutoff) && path != dir {
			err = os.RemoveAll(path)
			if err != nil {
				return err
			}
		}
		return nil
	})
}
