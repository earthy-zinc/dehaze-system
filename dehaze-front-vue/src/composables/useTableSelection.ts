import { ref } from "vue";

/**
 * 表格多选 Composable，管理已选中的行 ID 数组。
 *
 * @example
 * ```ts
 * const { selectedIds, handleSelectionChange, clearSelection, isSelected, toggleSelection } = useTableSelection();
 *
 * // 绑定到 Element Plus Table 的 selection-change 事件
 * <el-table :selection="selection" @selection-change="handleSelectionChange" />
 *
 * // 检查单个 ID 是否已选中
 * if (isSelected('some-id')) { ... }
 *
 * // 切换选中状态
 * toggleSelection('some-id');
 * ```
 *
 * @returns 选中状态与管理方法
 */
export function useTableSelection() {
  const selectedIds = ref<(string | number)[]>([]);

  /**
   * 处理 Element Plus Table 的 selection-change 事件，同步 selectedIds。
   * @param rows - 当前选中的行数据数组
   */
  const handleSelectionChange = (rows: any[]) => {
    selectedIds.value = rows
      .map((row) => row.id ?? row._id ?? null)
      .filter(Boolean) as (string | number)[];
  };

  /**
   * 清空所有选中。
   */
  const clearSelection = () => {
    selectedIds.value = [];
  };

  /**
   * 检查指定 ID 是否已选中。
   * @param id - 要检查的 ID
   * @returns 是否已选中
   */
  const isSelected = (id: string | number): boolean => {
    return selectedIds.value.includes(id);
  };

  /**
   * 切换指定 ID 的选中状态。
   * @param id - 要切换的 ID
   */
  const toggleSelection = (id: string | number) => {
    const index = selectedIds.value.indexOf(id);
    if (index > -1) {
      selectedIds.value.splice(index, 1);
    } else {
      selectedIds.value.push(id);
    }
  };

  return {
    selectedIds,
    handleSelectionChange,
    clearSelection,
    isSelected,
    toggleSelection,
  };
}
