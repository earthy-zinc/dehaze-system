import { MenuAPI, RoleAPI, type MenuVO } from "dehaze-sdk-js";
import { Button, Modal, Space, Spin, Tree, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState,
} from "react";

/** 递归转换菜单数据为 Tree 组件需要的格式 */
function buildMenuTree(menus: MenuVO[]): any[] {
  return menus.map((menu) => ({
    title: menu.name,
    key: menu.id,
    children: menu.children?.length ? buildMenuTree(menu.children) : undefined,
  }));
}

/** 递归收集所有树节点key */
function collectAllKeys(nodes: any[]): React.Key[] {
  const keys: React.Key[] = [];
  const walk = (list: any[]) => {
    list.forEach((n) => {
      keys.push(n.key);
      if (n.children?.length) walk(n.children);
    });
  };
  walk(nodes);
  return keys;
}

export interface PermissionDialogRef {
  open: (roleId: number, roleName: string) => void;
}

interface PermissionDialogProps {
  onSuccess?: () => void;
}

const PermissionDialog = forwardRef<PermissionDialogRef, PermissionDialogProps>(
  ({ onSuccess }, ref) => {
    const [visible, setVisible] = useState(false);
    const [confirmLoading, setConfirmLoading] = useState(false);
    const [loading, setLoading] = useState(false);
    const [roleId, setRoleId] = useState<number>(0);
    const [roleName, setRoleName] = useState("");
    const [menuTree, setMenuTree] = useState<any[]>([]);
    const [checkedKeys, setCheckedKeys] = useState<React.Key[]>([]);
    const [expandedKeys, setExpandedKeys] = useState<React.Key[]>([]);

    // 所有树节点key（用于全选/展开所有）
    const allKeys = useMemo(() => collectAllKeys(menuTree), [menuTree]);
    const isAllChecked =
      allKeys.length > 0 && checkedKeys.length >= allKeys.length;
    const isAllExpanded =
      allKeys.length > 0 && expandedKeys.length >= allKeys.length;

    const open = useCallback(async (id: number, name: string) => {
      setRoleId(id);
      setRoleName(name);
      setVisible(true);
      setLoading(true);

      try {
        const [menuData, menuIds] = await Promise.all([
          MenuAPI.getList({}),
          RoleAPI.getRoleMenuIds(id),
        ]);
        const treeData = buildMenuTree(menuData || []);
        setMenuTree(treeData);
        setCheckedKeys(menuIds || []);
        // 打开时默认展开所有节点
        setExpandedKeys(collectAllKeys(treeData));
      } catch {
        setMenuTree([]);
        setCheckedKeys([]);
        setExpandedKeys([]);
        message.error("加载权限数据失败");
      } finally {
        setLoading(false);
      }
    }, []);

    useImperativeHandle(ref, () => ({ open }), [open]);

    const handleCancel = useCallback(() => {
      setVisible(false);
    }, []);

    const handleOk = useCallback(async () => {
      setConfirmLoading(true);
      try {
        const ids = checkedKeys.map(Number);
        await RoleAPI.updateRoleMenus(roleId, ids);
        message.success(`角色「${roleName}」权限分配成功`);
        setVisible(false);
        onSuccess?.();
      } catch {
        message.error("权限分配失败");
      } finally {
        setConfirmLoading(false);
      }
    }, [roleId, roleName, checkedKeys, onSuccess]);

    const handleCheck = useCallback(
      (
        checked:
          | React.Key[]
          | { checked: React.Key[]; halfChecked: React.Key[] }
      ) => {
        const keys = Array.isArray(checked) ? checked : checked.checked;
        setCheckedKeys(keys);
      },
      []
    );

    /** 全选/取消全选 */
    const handleToggleCheckAll = useCallback(() => {
      setCheckedKeys(isAllChecked ? [] : allKeys);
    }, [isAllChecked, allKeys]);

    /** 展开/收起所有 */
    const handleToggleExpandAll = useCallback(() => {
      setExpandedKeys(isAllExpanded ? [] : allKeys);
    }, [isAllExpanded, allKeys]);

    return (
      <Modal
        title={`分配权限 - ${roleName}`}
        open={visible}
        width={500}
        confirmLoading={confirmLoading}
        okText="确定"
        cancelText="取消"
        destroyOnHidden
        onOk={handleOk}
        onCancel={handleCancel}
      >
        {loading ? (
          <div style={{ textAlign: "center", padding: 40 }}>
            <Spin />
          </div>
        ) : menuTree.length === 0 ? (
          <div style={{ textAlign: "center", padding: 40, color: "#999" }}>
            暂无菜单数据
          </div>
        ) : (
          <>
            {/* 工具栏：全选/取消全选、展开/收起所有 */}
            <Space style={{ marginBottom: 8 }}>
              <Button size="small" onClick={handleToggleCheckAll}>
                {isAllChecked ? "取消全选" : "全选"}
              </Button>
              <Button size="small" onClick={handleToggleExpandAll}>
                {isAllExpanded ? "收起所有" : "展开所有"}
              </Button>
            </Space>
            <Tree
              checkable
              expandedKeys={expandedKeys}
              onExpand={(keys) => setExpandedKeys(keys)}
              treeData={menuTree}
              checkedKeys={checkedKeys}
              onCheck={handleCheck}
            />
          </>
        )}
      </Modal>
    );
  }
);

PermissionDialog.displayName = "PermissionDialog";

export default PermissionDialog;
