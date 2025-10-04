package global

import (
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"github.com/redis/go-redis/v9"
	"github.com/songzhibin97/gkit/cache/local_cache"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

var (
	CONFIG Server

	VIPER    *viper.Viper
	VALIDATE *validator.Validate

	LOG *zap.Logger

	GIN    *gin.Engine
	ROUTES gin.RoutesInfo

	DB             *gorm.DB
	ACTIVE_DB_NAME *string

	LOCAL_CACHE local_cache.Cache
	REDIS       redis.UniversalClient
)
