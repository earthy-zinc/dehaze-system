package model

import "encoding/json"

// SysAiSkill Skill 主表（F-M08-006 Skills 管理）
type SysAiSkill struct {
	BaseModel
	Name          string          `gorm:"column:name;type:varchar(128);not null;comment:Skill名称(唯一)" json:"name"`
	Description   string          `gorm:"column:description;type:varchar(500);not null;comment:Skill描述" json:"description"`
	Scene         string          `gorm:"column:scene;type:varchar(255);not null;default:'';comment:适用场景" json:"scene"`
	Instruction   *string         `gorm:"column:instruction;type:text;comment:SKILL.md指令正文" json:"instruction"`
	License       *string         `gorm:"column:license;type:varchar(255);comment:frontmatter license" json:"license"`
	Compatibility *string         `gorm:"column:compatibility;type:varchar(500);comment:frontmatter compatibility" json:"compatibility"`
	Metadata      json.RawMessage `gorm:"column:metadata;type:json;comment:frontmatter metadata" json:"metadata"`
	AllowedTools  *string         `gorm:"column:allowed_tools;type:varchar(500);comment:frontmatter allowed-tools" json:"allowedTools"`
	// 不带 `default:` 标签：GORM 对带默认值的字段在零值时会**用默认值替换**（并省略列），
	// 导致"新建 Skill 默认禁用"（status=0，python `_STATUS_DISABLED`）永远插不进去 0。
	// 列本身的 DEFAULT 1 仍在（不显式给值时才生效），语义与 python 显式赋 0 一致。
	Status       int8   `gorm:"column:status;type:tinyint;not null;comment:启停状态(0:禁用;1:启用)" json:"status"`
	Source       string `gorm:"column:source;type:varchar(32);not null;default:admin;comment:来源(builtin/admin)" json:"source"`
	MarketShared int8   `gorm:"column:market_shared;type:tinyint;not null;default:0;comment:是否共享至市场" json:"marketShared"`
	Deleted      int64  `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiSkill) TableName() string {
	return "sys_ai_skill"
}

// SysAiSkillFile SKILL 目录文件清单（内容存对象存储）
type SysAiSkillFile struct {
	BaseModel
	SkillID  int64   `gorm:"column:skill_id;type:bigint;not null;comment:所属Skill主键" json:"skillId"`
	Path     string  `gorm:"column:path;type:varchar(500);not null;comment:相对SKILL根目录路径" json:"path"`
	FileSize int64   `gorm:"column:file_size;type:bigint;not null;default:0;comment:文件大小(字节)" json:"fileSize"`
	FileType *string `gorm:"column:file_type;type:varchar(64);comment:文件类型" json:"fileType"`
}

func (SysAiSkillFile) TableName() string {
	return "sys_ai_skill_file"
}
