package options

// AI AI 域转发配置（B 类端点：强依赖 deepagents/LLM/ES，行为在 dehaze-python）
type AI struct {
	ServiceURL     string `mapstructure:"serviceUrl" json:"serviceUrl" yaml:"serviceUrl"`             // Python 服务地址（与 algorithm 同源，AI 域转发目标）
	Timeout        int    `mapstructure:"timeout" json:"timeout" yaml:"timeout"`                      // JSON 转发总超时（秒），默认 120
	StreamTimeout  int    `mapstructure:"streamTimeout" json:"streamTimeout" yaml:"streamTimeout"`    // 流式转发空闲读超时（秒），默认 300
	ConnectTimeout int    `mapstructure:"connectTimeout" json:"connectTimeout" yaml:"connectTimeout"` // 连接超时（秒），默认 5
}
