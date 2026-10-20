/**
 * 样张静态资源基址。
 *
 * 来源：各环境 `.env` 的 `VITE_DATASET_HOST`（静态数据集服务，URL 前缀 `/datasets`）。
 * 开发环境见 `.env.development`，生产环境见 `.env.production`；本地临时覆盖写 `.env.development.local`
 * （不得在代码里写死存储域名，各环境域名不同）。
 */
const DATASET_BASE_URL = `${import.meta.env.VITE_DATASET_HOST}/datasets`;

// 静态源现仅有 001 这一对样张（NH-HAZE-2023），其余编号 404，故只保留可用样张
const exampleImages = [
  {
    id: 0,
    haze: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    clean: `${DATASET_BASE_URL}/NH-HAZE-2023/clean/001.JPG`,
  },
];

export default exampleImages;
