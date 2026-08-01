package api_key

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"math/big"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	apikeyrepo "github.com/earthyzinc/dehaze-go/internal/repository/api_key"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

type IApiKeyService interface {
	CreateApiKey(ctx context.Context, userID int64, req *dto.ApiKeyCreateRequest) (*dto.ApiKeyResult, error)
	ListApiKeys(ctx context.Context, userID int64) ([]dto.ApiKeyResult, error)
	Revoke(ctx context.Context, id int64, userID int64) error
	AuthenticateByKey(ctx context.Context, rawKey string) (*model.UserAuthInfo, error)
}

type ApiKeyService struct {
	apiKeyRepo  apikeyrepo.IApiKeyRepository
	userService userservice.IUserService
}

func NewApiKeyService(apiKeyRepo apikeyrepo.IApiKeyRepository, userService userservice.IUserService) *ApiKeyService {
	return &ApiKeyService{
		apiKeyRepo:  apiKeyRepo,
		userService: userService,
	}
}

func (s *ApiKeyService) CreateApiKey(ctx context.Context, userID int64, req *dto.ApiKeyCreateRequest) (*dto.ApiKeyResult, error) {
	rawKey, err := generateApiKey()
	if err != nil {
		return nil, common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "生成API Key失败")
	}

	keyHash := hashKey(rawKey)

	apiKey := &model.SysApiKey{
		UserID:    userID,
		Name:      req.Name,
		KeyPrefix: rawKey[:16],
		KeyHash:   keyHash,
		Status:    1,
		ExpiresAt: req.ExpiresAt,
	}
	apiKey.CreateBy = userID

	if err := s.apiKeyRepo.Create(ctx, apiKey); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建API Key失败", err)
	}

	return &dto.ApiKeyResult{
		ID:         apiKey.ID,
		Name:       apiKey.Name,
		ApiKey:     rawKey,
		KeyPrefix:  apiKey.KeyPrefix,
		Status:     apiKey.Status,
		ExpiresAt:  apiKey.ExpiresAt,
		CreateTime: apiKey.CreatedAt,
	}, nil
}

func (s *ApiKeyService) ListApiKeys(ctx context.Context, userID int64) ([]dto.ApiKeyResult, error) {
	keys, err := s.apiKeyRepo.FindByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询API Key列表失败", err)
	}

	results := make([]dto.ApiKeyResult, 0, len(keys))
	for _, k := range keys {
		results = append(results, dto.ApiKeyResult{
			ID:         k.ID,
			Name:       k.Name,
			KeyPrefix:  k.KeyPrefix,
			Status:     k.Status,
			ExpiresAt:  k.ExpiresAt,
			LastUsedAt: k.LastUsedAt,
			CreateTime: k.CreatedAt,
		})
	}
	return results, nil
}

func (s *ApiKeyService) Revoke(ctx context.Context, id int64, userID int64) error {
	if err := s.apiKeyRepo.RevokeByID(ctx, id, userID); err != nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "API Key不存在或无权操作")
	}
	return nil
}

func (s *ApiKeyService) AuthenticateByKey(ctx context.Context, rawKey string) (*model.UserAuthInfo, error) {
	keyHash := hashKey(rawKey)

	apiKey, err := s.apiKeyRepo.FindByHash(ctx, keyHash)
	if err != nil {
		return nil, err
	}
	if apiKey == nil {
		return nil, common.NewBizError(common.TOKEN_INVALID, "API Key无效")
	}
	if apiKey.RevokedAt != nil {
		return nil, common.NewBizError(common.TOKEN_INVALID, "API Key已被吊销")
	}
	if apiKey.ExpiresAt != nil && apiKey.ExpiresAt.Before(time.Now()) {
		return nil, common.NewBizError(common.TOKEN_INVALID, "API Key已过期")
	}

	authInfo, err := s.userService.GetUserAuthInfoByID(ctx, apiKey.UserID)
	if err != nil {
		return nil, err
	}

	_ = s.apiKeyRepo.UpdateLastUsed(ctx, apiKey.ID)

	return authInfo, nil
}

func generateApiKey() (string, error) {
	const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	b := make([]byte, 48)
	for i := range b {
		n, err := rand.Int(rand.Reader, big.NewInt(62))
		if err != nil {
			return "", err
		}
		b[i] = chars[n.Int64()]
	}
	return "dhak_" + string(b), nil
}

func hashKey(rawKey string) string {
	h := sha256.Sum256([]byte(rawKey))
	return hex.EncodeToString(h[:])
}
