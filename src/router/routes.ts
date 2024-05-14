export interface routeType {
  path: string;
  component?: any;
  children?: Array<routeType>;
  meta?: {
    title?: string;
    needLogin?: boolean;
  };
  redirect?: string;
}

const routes: Array<routeType> = [];
export default routes;
