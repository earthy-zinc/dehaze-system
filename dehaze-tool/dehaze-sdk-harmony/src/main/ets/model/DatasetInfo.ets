/**
 * 数据集管理数据模型
 *
 * 与后端 vo.DatasetVO / vo.ImageItem / vo.ImageUrl / model.SysItemFile 对齐
 */

/** 数据集查询参数 */
export class DatasetQuery {
  keywords?: string = undefined;
}

/** 数据集（后端 vo.DatasetVO） */
export class Dataset {
  id: number = 0;
  parentId: number = 0;
  type: string = '';
  name: string = '';
  description?: string = undefined;
  path: string = '';
  size: string = '';
  hasChildren: boolean = false;
  children?: Dataset[] = undefined;
  status: number = 1;
  total: number = 0;
  createTime?: string = undefined;
  updateTime?: string = undefined;
}

/** 数据项查询参数 */
export class DatasetItemQuery {
  keywords?: string = undefined;
  pageNum: number = 1;
  pageSize: number = 10;
  /** 数据集 ID */
  datasetId?: number = undefined;
}

/** 数据项 */
export class DatasetItem {
  id: number = 0;
  datasetId: number = 0;
  name?: string = undefined;
  /** 关联图片列表 */
  imageUrls?: ImageUrl[] = undefined;
  createTime?: string = undefined;
  updateTime?: string = undefined;
}

/** 图片 URL 信息（后端 vo.ImageUrl） */
export class ImageUrl {
  id: number = 0;
  /** 图片类型（clear 清晰图 / hazy 有雾图） */
  type: string = '';
  /** 图片 URL */
  url: string = '';
  /** 高清图片 URL */
  originUrl?: string = undefined;
  /** 描述 */
  description?: string = undefined;
}

/** 图片文件信息（后端 vo.ImageItem / model.SysItemFile） */
export class ImageItem {
  id: number = 0;
  /** 图片 URL 列表 */
  imgUrl: ImageUrl[] = [];
}

/** 配对图片上传参数 */
export class PairedImageUpload {
  /** 数据集 ID */
  datasetId: number = 0;
  /** 清晰图文件 URI */
  clearImageUri?: string = undefined;
  /** 有雾图文件 URI */
  hazyImageUri?: string = undefined;
  /** 描述 */
  description?: string = undefined;
}

/** 数据项图片文件信息（后端 dto.ImageFileInfo） */
export class DatasetImageFileInfo {
  id: number = 0;
  datasetItemId: number = 0;
  fileId: number = 0;
  type: string = '';
  description: string = '';
  url: string = '';
}
