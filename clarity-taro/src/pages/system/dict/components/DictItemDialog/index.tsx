import React from "react";
import { Icon, View, Text, Loading, Empty, Popup } from "@/platform/ui";
import { Button, SwipeCell, Cell } from "@taroify/core";
import type { DictPageVO } from "clarity-sdk-js";
import StatusTag from "@/components/common/StatusTag";

interface DictItemDialogProps {
  open: boolean;
  typeName: string;
  items: DictPageVO[];
  loading: boolean;
  canAdd: boolean;
  canEdit: boolean;
  canDelete: boolean;
  onClose: () => void;
  onAdd: () => void;
  onEdit: (id: number) => void;
  onDelete: (item: DictPageVO) => void;
}

const DictItemDialog: React.FC<DictItemDialogProps> = ({
  open,
  typeName,
  items,
  loading,
  canAdd,
  canEdit,
  canDelete,
  onClose,
  onAdd,
  onEdit,
  onDelete,
}) => {
  return (
    <Popup
      open={open}
      onClose={onClose}
      placement="bottom"
      style={{ height: "80%" }}
    >
      <View className="item-popup">
        <View className="item-header">
          <Text className="item-title">字典数据 - {typeName}</Text>
          {canAdd && (
            <Button size="small" color="primary" onClick={onAdd}>
              <Icon name="add" /> 新增
            </Button>
          )}
        </View>

        <View className="item-list">
          {loading && items.length === 0 ? (
            <Loading>加载中...</Loading>
          ) : items.length === 0 ? (
            <Empty>
              <Empty.Description>暂无字典数据</Empty.Description>
            </Empty>
          ) : (
            items.map((item) => (
              <SwipeCell key={item.id} className="item-swipe-cell">
                <SwipeCell.Actions side="right">
                  {canEdit && (
                    <Button size="small" onClick={() => onEdit(item.id!)}>
                      <Icon name="edit" /> 编辑
                    </Button>
                  )}
                  {canDelete && (
                    <Button
                      color="danger"
                      size="small"
                      onClick={() => onDelete(item)}
                    >
                      <Icon name="delete" /> 删除
                    </Button>
                  )}
                </SwipeCell.Actions>
                <Cell className="item-cell">
                  <View className="item-info">
                    <View className="item-name">{item.name}</View>
                    <View className="item-value">值: {item.value}</View>
                  </View>
                  <View className="item-status">
                    <StatusTag status={item.status} />
                  </View>
                </Cell>
              </SwipeCell>
            ))
          )}
        </View>

        {loading && items.length > 0 && (
          <View className="loading-more">
            <Loading size="small">加载中...</Loading>
          </View>
        )}
      </View>
    </Popup>
  );
};

export default DictItemDialog;
