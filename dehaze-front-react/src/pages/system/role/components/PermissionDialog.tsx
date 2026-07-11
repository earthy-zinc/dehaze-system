import { MenuAPI, RoleAPI, type MenuVO } from "dehaze-sdk-js";
import { Modal, Spin, Tree, message } from "antd";
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
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

export interface PermissionDialogRef {
  open: (roleId: number, roleName: string) => void;
}

interface PermissionDialogProps {
  onSuccess?: () => void;
}

const PermissionDialog = forwardRef<
  PermissionDialogRef,
  PermissionDialogProps
>(({ onSuccess }, ref) => {
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [roleId, setRoleId] = useState<number>(0);
  const [roleName, setRoleName] = useState("");
  const [menuTree, setMenuTree] = useState<any[]>([]);
  const [checkedKeys, setCheckedKeys] = useState<React.Key[]>([]);

  const open = useCallback(
    async (id: number, name: string) => {
      setRoleId(id);
      setRoleName(name);
      setVisible(true);
      setLoading(true);

      try {
        const [menuData, menuIds] = await Promise.all([
          MenuAPI.getList({}),
          RoleAPI.getRoleMenuIds(id),
        ]);
        setMenuTree(buildMenuTree(menuData || []));
        setCheckedKeys(menuIds || []);
      } catch {
        setMenuTree([]);
        setCheckedKeys([]);
        message.error("加载权限数据失败");
      } finally {
        setLoading(false);
      }
    },
    []
  );

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

  const handleCheck = useCallback((checked: React.Key[]) => {
    setCheckedKeys(checked);
  }, []);

  return (
    <Modal
      title={`分配权限 - ${roleName}`}
      open={visible}
      width={500}
      confirmLoading={confirmLoading}
      okText="确定"
      cancelText="取消"
      destroyOnClose
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
        <Tree
          checkable
          defaultExpandAll
          treeData={menuTree}
          checkedKeys={checkedKeys}
          onCheck={handleCheck}
        />
      )}
    </Modal>
  );
});

PermissionDialog.displayName = "PermissionDialog";

export default PermissionDialog;
