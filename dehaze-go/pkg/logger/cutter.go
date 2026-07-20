package logger

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Cutter struct {
	level        string
	layout       string
	formats      []string
	director     string
	retentionDay int
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
		if err := c.rotate(currentDate); err != nil {
			return 0, err
		}
	}

	return c.file.Write(bytes)
}

func (c *Cutter) rotate(currentDate string) error {
	if c.file != nil {
		if err := c.file.Close(); err != nil {
			return fmt.Errorf("关闭旧日志文件失败: %w", err)
		}
	}

	values := make([]string, 0, 3+len(c.formats))
	values = append(values, c.director)
	if c.layout != "" {
		values = append(values, currentDate)
	}
	values = append(values, c.formats...)
	values = append(values, c.level+".log")

	filename := filepath.Join(values...)
	director := filepath.Dir(filename)

	if err := os.MkdirAll(director, os.ModePerm); err != nil {
		return fmt.Errorf("创建日志目录失败: %w", err)
	}

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("打开日志文件失败: %w", err)
	}

	c.file = file
	c.currentDate = currentDate

	go c.cleanOldLogs()

	return nil
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
