export interface ApiKeyCreateForm {
  name: string;
  expiresAt?: string;
  dailyQuota?: number;
  monthlyQuota?: number;
  rpmLimit?: number;
  modelWhitelist?: string[];
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
  dailyQuota?: number;
  monthlyQuota?: number;
  rpmLimit?: number;
  modelWhitelist?: string[];
}
