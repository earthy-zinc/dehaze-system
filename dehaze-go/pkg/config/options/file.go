package options

// File 文件存储配置
// file.type 为默认存储后端标识（上传时使用）；各后端 baseUrl 必须是完整 URL（带 scheme+host），
// URL 运行时拼接为 storage.baseUrl + "/" + object_name，不落库。
type File struct {
	Type    string             `mapstructure:"type" json:"type" yaml:"type"` // 默认存储后端：minio/local/nginx-static
	MaxSize int64              `mapstructure:"maxSize" json:"maxSize" yaml:"maxSize"`
	Storage FileStorageConfig  `mapstructure:"storage" json:"storage" yaml:"storage"`
}

// FileStorageConfig 各存储后端配置
type FileStorageConfig struct {
	MinIO       FileMinIO       `mapstructure:"minio" json:"minio" yaml:"minio"`
	Local       FileLocal       `mapstructure:"local" json:"local" yaml:"local"`
	NginxStatic FileNginxStatic `mapstructure:"nginx-static" json:"nginx-static" yaml:"nginx-static"`
}

// FileMinIO MinIO 存储配置
// GetURL 直接使用 endpoint + bucketName 拼接 MinIO 直连 URL（bucket 已设为 public read）
type FileMinIO struct {
	Endpoint   string `mapstructure:"endpoint" json:"endpoint" yaml:"endpoint"`
	AccessKey  string `mapstructure:"accessKey" json:"accessKey" yaml:"accessKey"`
	SecretKey  string `mapstructure:"secretKey" json:"secretKey" yaml:"secretKey"`
	BucketName string `mapstructure:"bucketName" json:"bucketName" yaml:"bucketName"`
	BaseURL    string `mapstructure:"baseUrl" json:"baseUrl" yaml:"baseUrl"` // 已废弃，GetURL 不再使用
}

// FileLocal 本地存储配置
// BaseURL: 本地文件下载接口基础 URL（完整地址），如 http://host:port/api/v1/files/download
type FileLocal struct {
	UploadPath string `mapstructure:"uploadPath" json:"uploadPath" yaml:"uploadPath"`
	BaseURL    string `mapstructure:"baseUrl" json:"baseUrl" yaml:"baseUrl"`
}

// FileNginxStatic nginx 静态服务后端配置
// BaseURL: nginx 静态服务根地址（完整 URL，不带 /datasets 等资源子路径），如 http://host:9000
type FileNginxStatic struct {
	BaseURL string `mapstructure:"baseUrl" json:"baseUrl" yaml:"baseUrl"`
}
