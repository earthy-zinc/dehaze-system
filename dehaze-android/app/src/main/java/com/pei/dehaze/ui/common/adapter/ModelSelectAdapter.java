package com.pei.dehaze.ui.common.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.ai_model.AiModelVO;

import java.util.Objects;

/**
 * 去雾流程内联模型选择适配器（数据源：启用图像编辑模型列表）
 */
public class ModelSelectAdapter extends ListAdapter<AiModelVO, ModelSelectAdapter.ModelViewHolder> {

    public interface OnModelSelectedListener {
        void onModelSelected(AiModelVO model);
    }

    private final OnModelSelectedListener listener;
    private String selectedModelId;

    public ModelSelectAdapter(OnModelSelectedListener listener) {
        super(DIFF_CALLBACK);
        this.listener = listener;
    }

    private static final DiffUtil.ItemCallback<AiModelVO> DIFF_CALLBACK =
            new DiffUtil.ItemCallback<AiModelVO>() {
                @Override
                public boolean areItemsTheSame(@NonNull AiModelVO oldItem, @NonNull AiModelVO newItem) {
                    return Objects.equals(oldItem.getModelId(), newItem.getModelId());
                }

                @Override
                public boolean areContentsTheSame(@NonNull AiModelVO oldItem, @NonNull AiModelVO newItem) {
                    return Objects.equals(oldItem.getDisplayName(), newItem.getDisplayName())
                            && Objects.equals(oldItem.getModelType(), newItem.getModelType());
                }
            };

    public void setSelectedModelId(String modelId) {
        if (Objects.equals(selectedModelId, modelId)) {
            return;
        }
        selectedModelId = modelId;
        notifyItemRangeChanged(0, getItemCount());
    }

    @NonNull
    @Override
    public ModelViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_image_model, parent, false);
        return new ModelViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ModelViewHolder holder, int position) {
        holder.bind(getItem(position));
    }

    class ModelViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvName;
        private final TextView tvModelId;
        private final TextView tvSelected;

        ModelViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tv_model_name);
            tvModelId = itemView.findViewById(R.id.tv_model_id);
            tvSelected = itemView.findViewById(R.id.tv_model_selected);
        }

        void bind(AiModelVO model) {
            tvName.setText(model.getDisplayName() == null ? model.getModelId() : model.getDisplayName());
            tvModelId.setText(model.getModelId());
            boolean selected = Objects.equals(selectedModelId, model.getModelId());
            tvSelected.setVisibility(selected ? View.VISIBLE : View.GONE);
            itemView.setOnClickListener(v -> listener.onModelSelected(model));
        }
    }
}
