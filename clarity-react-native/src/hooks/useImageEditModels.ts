/**
 * 去雾（image_edit）模型选择
 *
 * 数据源为统一模型管理的启用模型列表；默认选中项依次取：调用方指定的模型
 * （路由参数 / 历史记录复用，须仍在启用列表内）、列表首项。
 * 用户偏好 defaultModel 属 chat 维度，不参与图像模型预选。
 */
import { useEffect, useState } from 'react';
import { AiModelAPI } from 'clarity-sdk-js';
import type { AiModelVO } from 'clarity-sdk-js';

export function useImageEditModels(preferredModelId?: string) {
  const [models, setModels] = useState<AiModelVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState<string>();

  useEffect(() => {
    let alive = true;
    setLoading(true);
    AiModelAPI.listEnabledModels('image_edit')
      .then(list => {
        if (!alive) return;
        setModels(list ?? []);
        setSelectedModelId(
          list?.find(m => m.modelId === preferredModelId)?.modelId ??
            list?.[0]?.modelId,
        );
      })
      .catch(() => {
        if (alive) setModels([]);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [preferredModelId]);

  return {
    models,
    loading,
    selectedModelId,
    selectedModel: models.find(m => m.modelId === selectedModelId) ?? null,
    selectModel: setSelectedModelId,
  };
}
