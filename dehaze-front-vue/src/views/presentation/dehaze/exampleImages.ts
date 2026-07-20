const DATASET_API_PREFIX = import.meta.env.VITE_DATASET_BASE_API || "/dataset-api";

const exampleImages = [
  {
    id: 0,
    haze: `${DATASET_API_PREFIX}/NH-HAZE-2023/hazy/001.JPG`,
    clean: `${DATASET_API_PREFIX}/NH-HAZE-2023/clean/001.JPG`,
  },
  {
    id: 1,
    haze: `${DATASET_API_PREFIX}/NH-HAZE-2023/hazy/002.JPG`,
    clean: `${DATASET_API_PREFIX}/NH-HAZE-2023/clean/002.JPG`,
  },
  {
    id: 2,
    haze: `${DATASET_API_PREFIX}/NH-HAZE-2023/hazy/003.JPG`,
    clean: `${DATASET_API_PREFIX}/NH-HAZE-2023/clean/003.JPG`,
  },
  {
    id: 3,
    haze: `${DATASET_API_PREFIX}/NH-HAZE-2023/hazy/004.JPG`,
    clean: `${DATASET_API_PREFIX}/NH-HAZE-2023/clean/004.JPG`,
  },
];

export default exampleImages;
