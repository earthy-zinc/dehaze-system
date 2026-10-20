import { useCallback, useState } from "react";
import type { TableProps } from "antd";

export interface UseTableSelectionReturn<T> {
  selectedRowKeys: React.Key[];
  selectedRows: T[];
  rowSelection: TableProps<T>["rowSelection"];
  clearSelection: () => void;
  isSelected: (key: React.Key) => boolean;
}

export function useTableSelection<T extends { id?: string | number }>(
  keyField: string = "id"
): UseTableSelectionReturn<T> {
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  const [selectedRows, setSelectedRows] = useState<T[]>([]);

  const clearSelection = useCallback(() => {
    setSelectedRowKeys([]);
    setSelectedRows([]);
  }, []);

  const isSelected = useCallback(
    (key: React.Key) => selectedRowKeys.includes(key),
    [selectedRowKeys]
  );

  const rowSelection: TableProps<T>["rowSelection"] = {
    selectedRowKeys,
    onChange: (newSelectedRowKeys: React.Key[], newSelectedRows: T[]) => {
      setSelectedRowKeys(newSelectedRowKeys);
      setSelectedRows(newSelectedRows);
    },
  };

  return {
    selectedRowKeys,
    selectedRows,
    rowSelection,
    clearSelection,
    isSelected,
  };
}
