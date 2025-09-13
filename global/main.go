package global

import (
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

var (
	CONFIG Server

	VIPER *viper.Viper
	LOG   *zap.Logger

	GIN    *gin.Engine
	ROUTES gin.RoutesInfo

	DB             *gorm.DB
	ACTIVE_DB_NAME *string

	REDIS redis.UniversalClient
)
