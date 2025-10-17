/**
 * HTTP管理器，用于处理网络请求
 */
export class HttpManager {
  private static instance: HttpManager = undefined;
  private baseUrl: string = '';
  private debug: boolean = false;

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
      throw new Error('HttpManager未初始化，请先调用initialize方法');
    }
    return HttpManager.instance;
  }

  /**
   * 发送GET请求
   * @param url 请求地址
   * @param params 请求参数
   * @returns Promise
   */
  public async get<T>(url: string, params?: Record<string, any>): Promise<T> {
    let fullUrl = this.baseUrl + url;
    
    // 添加查询参数
    if (params) {
      const queryParams = Object.keys(params)
        .filter(key => params[key] !== undefined && params[key] !== null)
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
        .join('&');
      if (queryParams) {
        fullUrl += (fullUrl.includes('?') ? '&' : '?') + queryParams;
      }
    }
    
    return this.request<T>('GET', fullUrl);
  }

  /**
   * 发送POST请求
   * @param url 请求地址
   * @param data 请求数据
   * @returns Promise
   */
  public async post<T>(url: string, data?: any): Promise<T> {
    const fullUrl = this.baseUrl + url;
    return this.request<T>('POST', fullUrl, data);
  }

  /**
   * 发送PUT请求
   * @param url 请求地址
   * @param data 请求数据
   * @returns Promise
   */
  public async put<T>(url: string, data?: any): Promise<T> {
    const fullUrl = this.baseUrl + url;
    return this.request<T>('PUT', fullUrl, data);
  }

  /**
   * 发送DELETE请求
   * @param url 请求地址
   * @returns Promise
   */
  public async delete<T>(url: string): Promise<T> {
    const fullUrl = this.baseUrl + url;
    return this.request<T>('DELETE', fullUrl);
  }

  /**
   * 发送PATCH请求
   * @param url 请求地址
   * @param data 请求数据
   * @returns Promise
   */
  public async patch<T>(url: string, data?: any): Promise<T> {
    const fullUrl = this.baseUrl + url;
    return this.request<T>('PATCH', fullUrl, data);
  }

  /**
   * 通用请求方法
   * @param method 请求方法
   * @param url 请求地址
   * @param data 请求数据
   * @returns Promise
   */
  private async request<T>(method: string, url: string, data?: any): Promise<T> {
    // 构建请求头
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };
    
    // 添加认证Token
    const token = TokenManager.getToken();
    if (token) {
      headers['Authorization'] = token;
    }
    
    // 打印调试信息
    if (this.debug) {
      console.log(`[HTTP] ${method} ${url}`);
      if (data) {
        console.log(`[HTTP] Request Data: ${JSON.stringify(data)}`);
      }
    }
    
    try {
      let response;
      if (data instanceof FormData) {
        // 处理表单数据
        headers['Content-Type'] = 'multipart/form-data';
        response = await fetch(url, {
          method: method,
          headers: {
            'Authorization': headers['Authorization'],
            'Accept': headers['Accept']
          },
          body: data
        });
      } else {
        // 处理JSON数据
        response = await fetch(url, {
          method: method,
          headers: headers,
          body: data ? JSON.stringify(data) : undefined
        });
      }
      
      // 打印响应调试信息
      if (this.debug) {
        console.log(`[HTTP] Response Status: ${response.status}`);
      }
      
      // 检查响应状态
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // 处理响应数据
      const result = await response.json();
      
      // 打印响应调试信息
      if (this.debug) {
        console.log(`[HTTP] Response Data: ${JSON.stringify(result)}`);
      }
      
      return result as T;
    } catch (error) {
      // 打印错误信息
      if (this.debug) {
        console.error(`[HTTP] Request failed: ${error}`);
      }
      
      throw error;
    }
  }
}