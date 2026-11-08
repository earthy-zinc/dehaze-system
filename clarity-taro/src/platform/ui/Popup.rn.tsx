import { Modal, Pressable, StyleSheet, type ViewStyle } from "react-native";

import type { PopupProps } from "./popup-shared";

const PLACEMENT_STYLE: Record<string, ViewStyle> = {
  bottom: { justifyContent: "flex-end" },
  top: { justifyContent: "flex-start" },
  center: { justifyContent: "center", alignItems: "center" },
  left: { justifyContent: "center", alignItems: "flex-start" },
  right: { justifyContent: "center", alignItems: "flex-end" },
};

/**
 * RN 端用 Modal 承载弹层：遮罩点击关闭，面板按 placement 定位。
 * 动画交给 Modal 的 slide，与 H5 的过渡观感接近。
 */
function Popup({
  open = false,
  placement = "bottom",
  rounded,
  onClose,
  style,
  children,
}: PopupProps) {
  return (
    <Modal
      visible={open}
      transparent
      animationType="slide"
      onRequestClose={() => (onClose as (() => void) | undefined)?.()}
      statusBarTranslucent
    >
      <Pressable
        style={[
          styles.backdrop,
          PLACEMENT_STYLE[placement] ?? PLACEMENT_STYLE.bottom,
        ]}
        onPress={() => (onClose as (() => void) | undefined)?.()}
      >
        {/* 阻止点击面板时冒泡到遮罩 */}
        <Pressable
          onPress={(event) => event.stopPropagation?.()}
          style={[
            styles.panel,
            rounded ? styles.rounded : null,
            style as ViewStyle,
          ]}
        >
          {children}
        </Pressable>
      </Pressable>
    </Modal>
  );
}

/** RN 端遮罩由 Modal 自带，这里保持 API 兼容即可 */
Popup.Backdrop = function Backdrop() {
  return null;
};

export default Popup as React.FC<PopupProps> & { Backdrop: React.FC };

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
  },
  panel: {
    width: "100%",
    backgroundColor: "#fff",
  },
  rounded: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    overflow: "hidden",
  },
});
