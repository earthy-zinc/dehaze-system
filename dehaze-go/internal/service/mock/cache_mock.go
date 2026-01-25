package mock

import (
	"context"
	"time"
)

type CacheMock struct {
	GetFunc    func(ctx context.Context, key string) (string, error)
	SetFunc    func(ctx context.Context, key string, value any, expiration time.Duration) error
	DeleteFunc func(ctx context.Context, keys ...string) error
	ExistsFunc func(ctx context.Context, key string) (bool, error)
	SetNXFunc  func(ctx context.Context, key string, value any, expiration time.Duration) (bool, error)
}

func (m *CacheMock) Get(ctx context.Context, key string) (string, error) {
	if m.GetFunc != nil {
		return m.GetFunc(ctx, key)
	}
	return "", nil
}

func (m *CacheMock) Set(ctx context.Context, key string, value any, expiration time.Duration) error {
	if m.SetFunc != nil {
		return m.SetFunc(ctx, key, value, expiration)
	}
	return nil
}

func (m *CacheMock) Delete(ctx context.Context, keys ...string) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, keys...)
	}
	return nil
}

func (m *CacheMock) Exists(ctx context.Context, key string) (bool, error) {
	if m.ExistsFunc != nil {
		return m.ExistsFunc(ctx, key)
	}
	return false, nil
}

func (m *CacheMock) SetNX(ctx context.Context, key string, value any, expiration time.Duration) (bool, error) {
	if m.SetNXFunc != nil {
		return m.SetNXFunc(ctx, key, value, expiration)
	}
	return false, nil
}
