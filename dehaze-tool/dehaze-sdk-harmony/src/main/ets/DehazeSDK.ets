import { TokenManager } from './utils/TokenManager';
import { HttpManager } from './utils/HttpManager';
import { UserAPI } from './api/user';
import { AuthAPI } from './api/auth';
import { AlgorithmAPI } from './api/algorithm';
import { DatasetAPI } from './api/dataset';
import { PredictionAPI } from './api/prediction';
import { EvaluationAPI } from './api/evaluation';
import { InputHistoryAPI } from './api/inputHistory';
import { DeptAPI } from './api/dept';
import { DictAPI } from './api/dict';
import { FileAPI } from './api/file';
import { MenuAPI } from './api/menu';
import { RoleAPI } from './api/role';
import { ApiKeyAPI } from './api/apiKey';
import { common } from '@kit.AbilityKit';

/**
 * DehazeSDK 主类，用于初始化和配置 API 客户端
 *
 * 使用方式：
 * 1. 在 EntryAbility.onCreate 中调用 DehazeSDK.initialize(context, builder)
 * 2. 通过 DehazeSDK.getInstance().authApi 等访问各 API
 * 3. 也可直接使用静态类：AuthAPI.login(...)
 */
export class DehazeSDK {
  private static instance: DehazeSDK = undefined;

  public authApi: AuthAPI = undefined;
  public userApi: UserAPI = undefined;
  public algorithmApi: AlgorithmAPI = undefined;
  public datasetApi: DatasetAPI = undefined;
  public predictionApi: PredictionAPI = undefined;
  public evaluationApi: EvaluationAPI = undefined;
  public inputHistoryApi: InputHistoryAPI = undefined;
  public deptApi: DeptAPI = undefined;
  public dictApi: DictAPI = undefined;
  public fileApi: FileAPI = undefined;
  public menuApi: MenuAPI = undefined;
  public roleApi: RoleAPI = undefined;
  public apiKeyApi: ApiKeyAPI = undefined;

  private constructor(builder: Builder) {
    // 初始化 HTTP 管理器
    HttpManager.initialize(builder.baseUrl, builder.debug);

    // 创建 API 服务实例
    this.authApi = new AuthAPI();
    this.userApi = new UserAPI();
    this.algorithmApi = new AlgorithmAPI();
    this.datasetApi = new DatasetAPI();
    this.predictionApi = new PredictionAPI();
    this.evaluationApi = new EvaluationAPI();
    this.inputHistoryApi = new InputHistoryAPI();
    this.deptApi = new DeptAPI();
    this.dictApi = new DictAPI();
    this.fileApi = new FileAPI();
    this.menuApi = new MenuAPI();
    this.roleApi = new RoleAPI();
    this.apiKeyApi = new ApiKeyAPI();
  }

  /**
   * 初始化 SDK
   *
   * 必须在 EntryAbility.onCreate 中调用，传入 AbilityContext 以初始化 TokenManager
   */
  public static async initialize(context: common.UIAbilityContext, builder: Builder): Promise<void> {
    if (DehazeSDK.instance === undefined) {
      DehazeSDK.instance = new DehazeSDK(builder);
    }
    // 初始化 Token 存储（preferences 需要 context）
    await TokenManager.initialize(context);
  }

  /** 获取单例 */
  public static getInstance(): DehazeSDK {
    if (DehazeSDK.instance === undefined) {
      throw new Error('DehazeSDK 未初始化，请先调用 initialize 方法');
    }
    return DehazeSDK.instance;
  }

  /** 设置 Token 失效回调（跳转登录页） */
  public static setTokenInvalidCallback(callback: () => void): void {
    HttpManager.getInstance().setTokenInvalidCallback(callback);
  }
}

/** SDK 构建器 */
export class Builder {
  public baseUrl: string = 'http://127.0.0.1:8989';
  public debug: boolean = false;

  public setBaseUrl(baseUrl: string): Builder {
    this.baseUrl = baseUrl;
    return this;
  }

  public setDebug(debug: boolean): Builder {
    this.debug = debug;
    return this;
  }
}
