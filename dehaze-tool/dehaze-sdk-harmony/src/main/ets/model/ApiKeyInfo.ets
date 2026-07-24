export class ApiKeyInfo {
  id: number = 0;
  name: string = '';
  apiKey?: string = undefined;
  keyPrefix: string = '';
  status: number = 1;
  expiresAt?: string = undefined;
  lastUsedAt?: string = undefined;
  createTime?: string = undefined;
}

export class ApiKeyCreateForm {
  name: string = '';
  expiresAt?: string = undefined;
}
