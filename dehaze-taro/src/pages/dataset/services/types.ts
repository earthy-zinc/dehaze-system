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
  image_type: "foggy" | "clear" | "annotated";
  width: number;
  height: number;
  file_size: number;
  tags: string;
  description: string;
  created_at: string;
}

export interface ApiResponse<T> {
  code: number;
  message?: string;
  data: T;
}

export interface PaginatedResponse<T> {
  list: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface DatasetListParams {
  page?: number;
  page_size?: number;
  search?: string;
}

export interface DatasetImageListParams {
  dataset_id: number;
  page?: number;
  page_size?: number;
  image_type?: "all" | "foggy" | "clear" | "annotated";
  search?: string;
}

export interface DatasetDetailParams {
  id: number;
}
