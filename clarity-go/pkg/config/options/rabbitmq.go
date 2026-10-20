package options

import "time"

// RabbitMQ RabbitMQ 配置
// url 示例: amqp://user:pass@host:5672/vhost
// exchangeType 默认 direct
// routingKeyPrefix 默认 task
// enabled 用于灰度切换
type RabbitMQ struct {
	Enabled          bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	URL              string `mapstructure:"url" json:"url" yaml:"url"`
	Exchange         string `mapstructure:"exchange" json:"exchange" yaml:"exchange"`
	ExchangeType     string `mapstructure:"exchangeType" json:"exchangeType" yaml:"exchangeType"`
	RoutingKeyPrefix string `mapstructure:"routingKeyPrefix" json:"routingKeyPrefix" yaml:"routingKeyPrefix"`

	// 重连策略配置
	// ReconnectMaxRetries 最大重试次数，0 表示无限重试（默认 0）
	ReconnectMaxRetries int `mapstructure:"reconnectMaxRetries" json:"reconnectMaxRetries" yaml:"reconnectMaxRetries"`
	// ReconnectInitialInterval 首次重连等待间隔（默认 1s）
	ReconnectInitialInterval time.Duration `mapstructure:"reconnectInitialInterval" json:"reconnectInitialInterval" yaml:"reconnectInitialInterval"`
	// ReconnectMaxInterval 退避上限间隔（默认 30s）
	ReconnectMaxInterval time.Duration `mapstructure:"reconnectMaxInterval" json:"reconnectMaxInterval" yaml:"reconnectMaxInterval"`
}
