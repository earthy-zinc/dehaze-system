package options

// File 文件存储配置
type File struct {
	Type    string  `mapstructure:"type" json:"type" yaml:"type"`       // 存储类型：minio/local
	BaseURL string  `mapstructure:"baseUrl" json:"baseUrl" yaml:"baseUrl"` // 文件访问基础URL
	MaxSize int64   `mapstructure:"maxSize" json:"maxSize" yaml:"maxSize"` // 单文件最大大小（字节），默认100MB
	MinIO   FileMinIO `mapstructure:"minio" json:"minio" yaml:"minio"`
	Local   FileLocal `mapstructure:"local" json:"local" yaml:"local"`
}

// FileMinIO MinIO 存储配置
type FileMinIO struct {
	Endpoint   string `mapstructure:"endpoint" json:"endpoint" yaml:"endpoint"`
	AccessKey  string `mapstructure:"accessKey" json:"accessKey" yaml:"accessKey"`
	SecretKey  string `mapstructure:"secretKey" json:"secretKey" yaml:"secretKey"`
	BucketName string `mapstructure:"bucketName" json:"bucketName" yaml:"bucketName"`
}

// FileLocal 本地存储配置
type FileLocal struct {
	UploadPath string `mapstructure:"uploadPath" json:"uploadPath" yaml:"uploadPath"`
}
