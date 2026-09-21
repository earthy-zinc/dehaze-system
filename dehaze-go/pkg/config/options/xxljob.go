package options

type XxlJob struct {
	Enabled     bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	ServerAddr  string `mapstructure:"server-addr" json:"serverAddr" yaml:"server-addr"`
	AccessToken string `mapstructure:"access-token" json:"accessToken" yaml:"access-token"`
	// 执行器注册地址（admin 回调用），留空由执行器库探测本机 IP；admin 在容器时由 run.py 注入 host.docker.internal
	ExecutorIp   string `mapstructure:"executor-ip" json:"executorIp" yaml:"executor-ip"`
	ExecutorPort string `mapstructure:"executor-port" json:"executorPort" yaml:"executor-port"`
	RegistryKey  string `mapstructure:"registry-key" json:"registryKey" yaml:"registry-key"`
}
