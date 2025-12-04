export interface Dataset {
  id: number;
  name: string;
  description: string;
  creator: string;
  thumbnail: string;
  total_images: number;
  foggy_count: number;
  clear_count: number;
  annotated_count: number;
  created_at: string;
  updated_at: string;
}

export interface DatasetImage {
  id: number;
  dataset_id: number;
  filename: string;
  image_url: string;
  image_type: 'foggy' | 'clear' | 'annotated';
  width: number;
  height: number;
  file_size: number;
  tags: string;
  description: string;
  created_at: string;
}

export interface DatasetListResponse {
  code: number;
  data: {
    list: Dataset[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  };
}

export interface DatasetDetailResponse {
  code: number;
  data: Dataset;
}

export interface ImageListResponse {
  code: number;
  data: {
    list: DatasetImage[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  };
}

export type ImageTypeFilter = 'all' | 'foggy' | 'clear' | 'annotated';
export type ViewMode = 'list' | 'detail';