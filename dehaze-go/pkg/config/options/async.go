package options

// WorkerPool 线程池配置
type WorkerPool struct {
	CorePoolSize int    `mapstructure:"corePoolSize" json:"corePoolSize" yaml:"corePoolSize"` // 核心线程数
	MaxPoolSize  int    `mapstructure:"maxPoolSize" json:"maxPoolSize" yaml:"maxPoolSize"`    // 最大线程数
	QueueSize    int    `mapstructure:"queueSize" json:"queueSize" yaml:"queueSize"`          // 队列大小
	TaskTimeout  int    `mapstructure:"taskTimeout" json:"taskTimeout" yaml:"taskTimeout"`    // 任务超时时间(秒)
	ShutdownWait int    `mapstructure:"shutdownWait" json:"shutdownWait" yaml:"shutdownWait"` // 关闭等待时间(秒)
	ThreadName   string `mapstructure:"threadName" json:"threadName" yaml:"threadName"`       // 线程名称前缀
}

// AsyncTask 异步任务配置
type AsyncTask struct {
	DatasetTask WorkerPool `mapstructure:"datasetTask" json:"datasetTask" yaml:"datasetTask"` // 数据集任务线程池
	ImageTask   WorkerPool `mapstructure:"imageTask" json:"imageTask" yaml:"imageTask"`       // 图片任务线程池
	ExportTask  WorkerPool `mapstructure:"exportTask" json:"exportTask" yaml:"exportTask"`    // 导出任务线程池
}
