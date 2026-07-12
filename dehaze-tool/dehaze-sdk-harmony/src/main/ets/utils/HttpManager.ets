import { http } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { Result, ResultCode, TOKEN_INVALID_CODES } from '../model/Result';
import { TokenManager } from './TokenManager';

/** 请求方法类型 */
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

/** 请求选项 */
export interface RequestOptions {
  url: string;
  method?: HttpMethod;
  /** 请求数据（GET 作为 query 参数，其余作为 JSON body） */
  data?: Record<string, Object> | string | ArrayBuffer;
  /** 额外请求头 */
  headers?: Record<string, string>;
  /** 超时时间（毫秒），默认 30000 */
  timeout?: number;
  /** 是否跳过 Token 注入（登录等公开接口） */
  skipAuth?: boolean;
}

/** 文件上传选项 */
export interface UploadOptions {
  url: string;
  /** 文件 URI（如内部 URI 或沙箱路径） */
  fileUri: string;
  /** 文件字段名，默认 "file" */
  name?: string;
  /** 附加表单数据 */
  formData?: Record<string, string>;
  /** 额外请求头 */
  headers?: Record<string, string>;
  /** 超时时间（毫秒） */
  timeout?: number;
}

/**
 * HTTP 管理器
 *
 * 基于 @kit.NetworkKit 的 http 模块实现，提供：
 * - 请求拦截器（自动注入 Authorization 头）
 * - 响应拦截器（统一业务码处理）
 * - Token 失效回调（供上层跳转登录）
 * - 文件上传（multipart/form-data）
 */
export class HttpManager {
  private static instance: HttpManager = undefined;
  private baseUrl: string = '';
  private debug: boolean = false;

  /** Token 失效回调，由应用层注册（跳转登录页） */
  public onTokenInvalid: (() => void) | null = null;

  private constructor(baseUrl: string, debug: boolean) {
    this.baseUrl = baseUrl;
    this.debug = debug;
  }

  /** 初始化（仅一次） */
  public static initialize(baseUrl: string, debug: boolean): void {
    if (HttpManager.instance === undefined) {
      HttpManager.instance = new HttpManager(baseUrl, debug);
    }
  }

  /** 获取单例 */
  public static getInstance(): HttpManager {
    if (HttpManager.instance === undefined) {
      throw new Error('HttpManager 未初始化，请先调用 initialize 方法');
    }
    return HttpManager.instance;
  }

  /** 设置 Token 失效回调 */
  public setTokenInvalidCallback(callback: () => void): void {
    this.onTokenInvalid = callback;
  }

  /** GET 请求 */
  public async get<T>(url: string, params?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'GET', data: params });
  }

  /** POST 请求 */
  public async post<T>(url: string, data?: Record<string, Object> | string | ArrayBuffer): Promise<T> {
    return this.request<T>({ url, method: 'POST', data });
  }

  /** PUT 请求 */
  public async put<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'PUT', data });
  }

  /** DELETE 请求 */
  public async delete<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'DELETE', data });
  }

  /** PATCH 请求 */
  public async patch<T>(url: string, data?: Record<string, Object>): Promise<T> {
    return this.request<T>({ url, method: 'PATCH', data });
  }

  /**
   * 文件上传（multipart/form-data）
   * 返回后端统一响应结构解析后的 data
   */
  public async upload<T>(options: UploadOptions): Promise<T> {
    const fullUrl = this.baseUrl + options.url;
    const headers: Record<string, string> = {
      'Accept': 'application/json',
      ...options.headers,
    };

    // 注入 Token
    const token = TokenManager.getToken();
    if (token) {
      headers['Authorization'] = token.startsWith('Bearer ') ? token : `Bearer ${token}`;
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

  /** 通用请求方法 */
  public async request<T>(options: RequestOptions): Promise<T> {
    const { url, method = 'GET', data, headers = {}, timeout = 30000, skipAuth = false } = options;

    // 构建完整 URL（GET 请求拼接 query 参数）
    let fullUrl = this.baseUrl + url;
    let body: string | undefined = undefined;

    const reqHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...headers,
    };

    // 注入 Token
    if (!skipAuth) {
      const token = TokenManager.getToken();
      if (token) {
        reqHeaders['Authorization'] = token.startsWith('Bearer ') ? token : `Bearer ${token}`;
      }
    }

    // 处理请求数据
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

    // 发起请求
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

        // HTTP 状态码非 200
        if (response.responseCode !== 200) {
          reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, `HTTP ${response.responseCode}`));
          return;
        }

        // 解析响应体
        let result: Result<T>;
        try {
          result = JSON.parse(response.result as string) as Result<T>;
        } catch (e) {
          reject(new ApiError(ResultCode.SYSTEM_EXECUTION_ERROR, '响应解析失败'));
          return;
        }

        // 业务成功
        if (result.isSuccess()) {
          resolve(result.data);
          return;
        }

        // Token 失效
        if (TOKEN_INVALID_CODES.includes(result.code)) {
          TokenManager.clearToken();
          if (this.onTokenInvalid) {
            this.onTokenInvalid();
          }
          reject(new ApiError(result.code, result.msg));
          return;
        }

        // 业务错误
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

  /** 获取网络错误提示消息 */
  private getErrorMessage(err: BusinessError): string {
    const code = err.code;
    if (code === 2300001) return '网络不可用，请检查网络连接';
    if (code === 2300006) return '请求超时，请重试';
    if (code === 2300009) return '无法连接到服务器，请稍后重试';
    return err.message || '请求失败，请重试';
  }
}

/** API 错误类 */
export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = 'ApiError';
  }
}
