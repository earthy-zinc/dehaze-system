import { TokenManager } from './utils/TokenManager';
import { HttpManager } from './utils/HttpManager';
import { UserAPI } from './api/user';
import { AuthAPI } from './api/auth';
import { AlgorithmAPI } from './api/algorithm';
import { DatasetAPI } from './api/dataset';
import { DeptAPI } from './api/dept';
import { DictAPI } from './api/dict';
import { FileAPI } from './api/file';
import { MenuAPI } from './api/menu';
import { ModelAPI } from './api/model';
import { RoleAPI } from './api/role';

/**
 * SDK主类，用于初始化和配置API客户端
 */
export class DehazeSDK {
  private static instance: DehazeSDK = undefined;

  public authApi: AuthAPI = undefined;
  public userApi: UserAPI = undefined;
  public algorithmApi: AlgorithmAPI = undefined;
  public datasetApi: DatasetAPI = undefined;
  public deptApi: DeptAPI = undefined;
  public dictApi: DictAPI = undefined;
  public fileApi: FileAPI = undefined;
  public menuApi: MenuAPI = undefined;
  public modelApi: ModelAPI = undefined;
  public roleApi: RoleAPI = undefined;

  private constructor(builder: Builder) {
    const baseUrl = builder.baseUrl;
    const debug = builder.debug;

    // 初始化HTTP管理器
    HttpManager.initialize(baseUrl, debug);

    // 创建API服务实例
    this.authApi = new AuthAPI();
    this.userApi = new UserAPI();
    this.algorithmApi = new AlgorithmAPI();
    this.datasetApi = new DatasetAPI();
    this.deptApi = new DeptAPI();
    this.dictApi = new DictAPI();
    this.fileApi = new FileAPI();
    this.menuApi = new MenuAPI();
    this.modelApi = new ModelAPI();
    this.roleApi = new RoleAPI();
  }

  public static getInstance(): DehazeSDK {
    if (DehazeSDK.instance === undefined) {
      throw new Error('DehazeSDK未初始化，请先调用initialize方法');
    }
    return DehazeSDK.instance;
  }

  public static initialize(builder: Builder): void {
    DehazeSDK.instance = new DehazeSDK(builder);
  }
}

export class Builder {
  public baseUrl: string = 'http://localhost:8989';
  public debug: boolean = false;

  public setBaseUrl(baseUrl: string): Builder {
    this.baseUrl = baseUrl;
    return this;
  }

  public setDebug(debug: boolean): Builder {
    this.debug = debug;
    return this;
  }

  public build(): DehazeSDK {
    return new DehazeSDK(this);
  }
}