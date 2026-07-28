package options

type Mongo struct {
	URI      string `mapstructure:"uri" json:"uri" yaml:"uri"`
	Database string `mapstructure:"database" json:"database" yaml:"database"`
}
