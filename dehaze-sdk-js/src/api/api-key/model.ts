export interface ApiKeyCreateForm {
  name: string;
  expiresAt?: string;
}

export interface ApiKeyVO {
  id: number;
  name: string;
  apiKey?: string;
  keyPrefix: string;
  status: number;
  expiresAt?: string;
  lastUsedAt?: string;
  createTime?: string;
}
