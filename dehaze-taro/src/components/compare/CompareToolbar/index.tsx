import React, { useCallback } from "react";
import { View, Text } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { saveImageToAlbum } from "@/utils/saveImage";
import { COMPARE_MODES, type CompareMode } from "../types";
import "./index.less";

interface CompareToolbarProps {
  /** 当前激活的模式 */
  currentMode: CompareMode;
  /** 结果图 URL（用于保存/分享） */
  resultUrl?: string;
}

/**
 * 对比模式通用工具栏
 * 包含模式切换标签 + 4 个操作按钮（保存/分享/重新处理/换算法）
 */
const CompareToolbar: React.FC<CompareToolbarProps> = ({
  currentMode,
  resultUrl,
}) => {
  // 切换对比模式（统一使用 redirectTo 避免页面栈堆积）
  const handleSwitchMode = useCallback(
    (mode: CompareMode, path: string) => {
      if (mode === currentMode) return;
      Taro.redirectTo({ url: path });
    },
    [currentMode]
  );

  // 保存结果到相册（先下载到本地临时路径，再保存）
  const handleSave = useCallback(async () => {
    if (!resultUrl) {
      Taro.showToast({ title: "无结果图片可保存", icon: "none" });
      return;
    }
    await saveImageToAlbum(resultUrl, { h5Download: true });
  }, [resultUrl]);

  // 分享图片
  const handleShare = useCallback(() => {
    if (!resultUrl) {
      Taro.showToast({ title: "无结果图片可分享", icon: "none" });
      return;
    }
    if (process.env.TARO_ENV === "h5") {
      // H5 端：优先使用 Web Share API，失败则在新标签页打开
      if (typeof navigator !== "undefined" && navigator.share) {
        navigator.share({ title: "去雾结果", url: resultUrl }).catch(() => {
          window.open(resultUrl, "_blank");
        });
      } else {
        window.open(resultUrl, "_blank");
      }
      return;
    }
    Taro.showShareImageMenu({
      path: resultUrl,
      fail: () => Taro.showToast({ title: "分享失败", icon: "none" }),
    });
  }, [resultUrl]);

  // 重新处理
  const handleReprocess = useCallback(() => {
    Taro.redirectTo({ url: "/pages/processing/index" });
  }, []);

  // 更换算法
  const handleChangeAlgorithm = useCallback(() => {
    Taro.redirectTo({ url: "/pages/algorithm-select/index" });
  }, []);

  const actions = [
    { icon: "💾", text: "保存", onClick: handleSave },
    { icon: "📤", text: "分享", onClick: handleShare },
    { icon: "🔄", text: "重新处理", onClick: handleReprocess },
    { icon: "⚡", text: "换算法", onClick: handleChangeAlgorithm },
  ];

  return (
    <View className="compare-toolbar">
      {/* 第一行：模式切换 */}
      <View className="mode-tabs">
        {COMPARE_MODES.map((mode) => (
          <View
            key={mode.key}
            className={`mode-tab ${mode.key === currentMode ? "active" : ""}`}
            onClick={() => handleSwitchMode(mode.key, mode.path)}
          >
            <Text>{mode.label}</Text>
          </View>
        ))}
      </View>

      {/* 第二行：操作按钮 */}
      <View className="action-buttons">
        {actions.map((action) => (
          <View
            key={action.text}
            className="action-btn"
            onClick={action.onClick}
          >
            <Text className="action-icon">{action.icon}</Text>
            <Text className="action-text">{action.text}</Text>
          </View>
        ))}
      </View>
    </View>
  );
};

export default CompareToolbar;
