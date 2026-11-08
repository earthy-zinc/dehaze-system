import {
  Add,
  Arrow,
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  BarChartOutlined,
  BrushOutlined,
  BulbOutlined,
  CalendarOutlined,
  ChartTrendingOutlined,
  Clock,
  ClockOutlined,
  Close,
  Cross,
  Delete,
  DeleteOutlined,
  Description,
  DescriptionOutlined,
  Edit,
  Exchange,
  Expand,
  ExpandOutlined,
  FilterOutlined,
  Fire,
  FlagOutlined,
  Gift,
  GoodJobOutlined,
  Lock,
  ManagerOutlined,
  Phone,
  PhoneOutlined,
  Photo,
  PhotoOutlined,
  Photograph,
  Plus,
  QuestionOutlined,
  Search,
  SettingOutlined,
  Star,
  StarOutlined,
  Success,
  WarningOutlined,
} from "@taroify/icons";

import type { IconName } from "./icon-map";
import type { IconProps } from "./icon-shared";

/**
 * 小程序与 H5 的图标：直接用 Taroify 的组件（iconfont + CSS 伪元素）。
 * RN 端走同目录的 Icon.rn.tsx，用同一份字体的码点渲染。
 */
const COMPONENTS = {
  "arrow-left": ArrowLeft,
  arrow: Arrow,
  "arrow-up": ArrowUp,
  "arrow-down": ArrowDown,
  edit: Edit,
  delete: Delete,
  add: Add,
  plus: Plus,
  star: Star,
  "star-o": StarOutlined,
  "photo-o": PhotoOutlined,
  photo: Photo,
  search: Search,
  photograph: Photograph,
  lock: Lock,
  gift: Gift,
  fire: Fire,
  cross: Cross,
  close: Close,
  success: Success,
  "setting-o": SettingOutlined,
  phone: Phone,
  "phone-o": PhoneOutlined,
  expand: Expand,
  "expand-o": ExpandOutlined,
  "delete-o": DeleteOutlined,
  "bulb-o": BulbOutlined,
  "bar-chart-o": BarChartOutlined,
  "brush-o": BrushOutlined,
  "calendar-o": CalendarOutlined,
  "chart-trending-o": ChartTrendingOutlined,
  clock: Clock,
  "clock-o": ClockOutlined,
  description: Description,
  "description-o": DescriptionOutlined,
  exchange: Exchange,
  "good-job-o": GoodJobOutlined,
  "manager-o": ManagerOutlined,
  "question-o": QuestionOutlined,
  "warning-o": WarningOutlined,
  "flag-o": FlagOutlined,
  "filter-o": FilterOutlined,
} as const;

export type { IconProps } from "./icon-shared";

export default function Icon({
  name,
  ...rest
}: IconProps & { name: IconName }) {
  const Component = COMPONENTS[name as keyof typeof COMPONENTS];
  if (!Component) return null;
  return <Component {...rest} />;
}
