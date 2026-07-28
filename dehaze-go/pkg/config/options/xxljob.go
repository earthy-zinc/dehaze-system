package options

type XxlJob struct {
	Enabled      bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	ServerAddr   string `mapstructure:"server-addr" json:"serverAddr" yaml:"server-addr"`
	AccessToken  string `mapstructure:"access-token" json:"accessToken" yaml:"access-token"`
	ExecutorPort string `mapstructure:"executor-port" json:"executorPort" yaml:"executor-port"`
	RegistryKey  string `mapstructure:"registry-key" json:"registryKey" yaml:"registry-key"`
}
