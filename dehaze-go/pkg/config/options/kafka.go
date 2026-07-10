package options

// Kafka Kafka 配置（用于日志投递）
// enabled 为开关
// brokers 例: ["host1:9092","host2:9092"]
type Kafka struct {
	Enabled  bool     `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	Brokers  []string `mapstructure:"brokers" json:"brokers" yaml:"brokers"`
	Topic    string   `mapstructure:"topic" json:"topic" yaml:"topic"`
	ClientID string   `mapstructure:"clientId" json:"clientId" yaml:"clientId"`
}
