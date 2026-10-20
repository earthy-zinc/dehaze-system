package options

type Session struct {
	Cookie Cookie `mapstructure:"cookie" json:"cookie" yaml:"cookie"`
}

type Cookie struct {
	Secure bool   `mapstructure:"secure" json:"secure" yaml:"secure"`
	Path   string `mapstructure:"path" json:"path" yaml:"path"`
}
