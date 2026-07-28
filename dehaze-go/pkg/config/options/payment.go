package options

type Payment struct {
	Wechat WechatPayConfig `mapstructure:"wechat" json:"wechat" yaml:"wechat"`
	Alipay AlipayConfig    `mapstructure:"alipay" json:"alipay" yaml:"alipay"`
}

type WechatPayConfig struct {
	Enabled      bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	AppID        string `mapstructure:"appId" json:"appId" yaml:"appId"`
	MchID        string `mapstructure:"mchId" json:"mchId" yaml:"mchId"`
	MchKey       string `mapstructure:"mchKey" json:"mchKey" yaml:"mchKey"`
	NotifyURL    string `mapstructure:"notifyUrl" json:"notifyUrl" yaml:"notifyUrl"`
	APIKey       string `mapstructure:"apiKey" json:"apiKey" yaml:"apiKey"`
	APIClientKey string `mapstructure:"apiClientKey" json:"apiClientKey" yaml:"apiClientKey"`
	BaseURL      string `mapstructure:"baseUrl" json:"baseUrl" yaml:"baseUrl"`
}

type AlipayConfig struct {
	Enabled   bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`
	AppID     string `mapstructure:"appId" json:"appId" yaml:"appId"`
	PrivateKey string `mapstructure:"privateKey" json:"privateKey" yaml:"privateKey"`
	PublicKey  string `mapstructure:"publicKey" json:"publicKey" yaml:"publicKey"`
	NotifyURL  string `mapstructure:"notifyUrl" json:"notifyUrl" yaml:"notifyUrl"`
	Gateway    string `mapstructure:"gateway" json:"gateway" yaml:"gateway"`
}
