/**
 * 模型查询参数类型
 */
export class AlgorithmQuery {
  keywords?: string = undefined;
}

export class Algorithm {
  id: number = 0;
  parentId: number = 0;
  name: string = '';
  type: string = '';
  description: string = '';
  img?: string = undefined;
  path?: string = undefined;
  importPath?: string = undefined;
  params?: string = undefined;
  flops?: string = undefined;
  status?: number = undefined;
  size?: string = undefined;
  children?: Array<Algorithm> = undefined;
}