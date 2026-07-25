import { http } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { Result, ResultCode, TOKEN_INVALID_CODES } from '../model/Result';
import { TokenManager } from './TokenManager';

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface RequestOptions {
  url: string;
  method?: HttpMethod;
  data?: Record<string, Object> | string | ArrayBuffer;
  headers?: Record<string, string>;
  timeout?: number;
  skipAuth?: boolean;
}

export interface UploadOptions {
  url: string;
  fileUri: string;
  name?: string;
  formData?: Record<string, string>;
  headers?: Record<string, string>;
  timeout?: number;
}

export class HttpManager {
  private static instance: HttpManager = undefined;
  private baseUrl: string = '';
  private debug: boolean = false;

  public onTokenInvalid: (() => void) | null = null;

  private constructor(baseUrl: string, debug: boolean) {
    this.baseUrl = baseUrl;
    this.debug = debug;
  }

  public static initialize(baseUrl: string, debug: boolean): void {
    if (HttpManager.instance === undefined) {
      HttpManager.instance = new HttpManager(baseUrl, debug);
    }
  }

  public static getInstance(): HttpManager {
    if (HttpManager.instance === undefined) {
      throw new Error('HttpManager 未初始化，请先调用 initialize 方法');
    }
    return HttpManager.instance;
  }

  public setTokenInvalidCallback(callback: () => void): void {
    this.onTokenInvalid = callback;
  }

  public async get<T>(url: string, params?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'GET', data: params });
  }

  public async post<T>(url: string, data?: Record<string, Object> | string | ArrayBuffer): Promise<T> {
    return this.request<T>({ url, method: 'POST', data });
  }

  public async put<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'PUT', data });
  }

  public async delete<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'DELETE', data });
  }

  public async patch<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'PATCH', data });
  }

  public async upload<T>(options: UploadOptions): Promise<T> {
    const fullUrl = this.baseUrl + options.url;
    const headers: Record<string, string> = {
      'Accept': 'application/json',
      ...options.headers,
    };

    const sid = TokenManager.getSessionId();
    if (sid) {
      headers['X-Session-Id'] = sid;
    }

    const formData: Record<string, string> = { ...options.formData };

    if (this.debug) {
      console.log(`[HTTP-Upload] POST ${fullUrl}`);
    }

    return new Promise<T>((resolve, reject) => {
      const uploadTask = http.createUpload();
      uploadTask.on('progress', (data) => {
        if (this.debug) {
          console.log(`[HTTP-Upload] progress: ${data.processedSize}/${data.totalSize}`);
        }
      });
      uploadTask.upload({
        url: fullUrl,
        header: headers,
        method: http.RequestMethod.POST,
        files: [{ filename: options.name || 'file', name: options.name || 'file', uri: options.fileUri }],
        data: Object.keys(formData).map(key => ({ name: key, value: formData[key] })),
      }).then((uploadResult) => {
        try {
          const response = JSON.parse(uploadResult.body) as Result<T>;
          if (this.debug) {
            console.log(`[HTTP-Upload] Response: ${JSON.stringify(response)}`);
          }
          if (response.isSuccess()) {
            resolve(response.data);
          } else {
            reject(new ApiError(response.code, response.msg));
          }
        } catch (e) {
          reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, '解析上传响应失败'));
        }
      }).catch((err: BusinessError) => {
        if (this.debug) {
          console.error(`[HTTP-Upload] failed: ${JSON.stringify(err)}`);
        }
        reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, `上传失败: ${err.message}`));
      });
    });
  }

  public async request<T>(options: RequestOptions): Promise<T> {
    const { url, method = 'GET', data, headers = {}, timeout = 30000, skipAuth = false } = options;

    let fullUrl = this.baseUrl + url;
    let body: string | undefined = undefined;

    const reqHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...headers,
    };

    if (!skipAuth) {
      const sid = TokenManager.getSessionId();
      if (sid) {
        reqHeaders['X-Session-Id'] = sid;
      }
    }

    if (method === 'GET' && data && typeof data === 'object') {
      const queryParams = Object.keys(data)
        .filter(key => (data as Record<string, Object>)[key] !== undefined && (data as Record<string, Object>)[key] !== null)
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(String((data as Record<string, Object>)[key]))}`)
        .join('&');
      if (queryParams) {
        fullUrl += (fullUrl.includes('?') ? '&' : '?') + queryParams;
      }
    } else if (data) {
      body = typeof data === 'string' ? data : JSON.stringify(data);
    }

    if (this.debug) {
      console.log(`[HTTP] ${method} ${fullUrl}`);
      if (body) {
        console.log(`[HTTP] Body: ${body}`);
      }
    }

    return new Promise<T>((resolve, reject) => {
      const httpRequest = http.createHttp();

      httpRequest.request(fullUrl, {
        method: method as http.RequestMethod,
        header: reqHeaders,
        extraData: body,
        expectDataType: http.HttpDataType.STRING,
        connectTimeout: timeout,
        readTimeout: timeout,
      }).then((response) => {
        if (this.debug) {
          console.log(`[HTTP] ResponseCode: ${response.responseCode}`);
          console.log(`[HTTP] Response: ${response.result}`);
        }

        if (response.responseCode !== 200) {
          reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, `HTTP ${response.responseCode}`));
          return;
        }

        let result: Result<T>;
        try {
          result = JSON.parse(response.result as string) as Result<T>;
        } catch (e) {
          reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, '响应解析失败'));
          return;
        }

        if (result.isSuccess()) {
          resolve(result.data);
          return;
        }

        if (TOKEN_INVALID_CODES.includes(result.code)) {
          TokenManager.clearToken();
          if (this.onTokenInvalid) {
            this.onTokenInvalid();
          }
          reject(new ApiError(result.code, result.msg));
          return;
        }

        reject(new ApiError(result.code, result.msg));
      }).catch((err: BusinessError) => {
        if (this.debug) {
          console.error(`[HTTP] Request failed: ${JSON.stringify(err)}`);
        }
        const message = this.getErrorMessage(err);
        reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, message));
      }).finally(() => {
        httpRequest.destroy();
      });
    });
  }

  private getErrorMessage(err: BusinessError): string {
    const code = err.code;
    if (code === 2300001) return '网络不可用，请检查网络连接';
    if (code === 2300006) return '请求超时，请重试';
    if (code === 2300009) return '无法连接到服务器，请稍后重试';
    return err.message || '请求失败，请重试';
  }
}

export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = 'ApiError';
  }
}
