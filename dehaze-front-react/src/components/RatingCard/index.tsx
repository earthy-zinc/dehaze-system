import { FeedbackAPI, FileAPI } from "dehaze-sdk-js";
import { PictureOutlined, PlusOutlined } from "@ant-design/icons";
import {
  Form,
  Input,
  message,
  Modal,
  Rate,
  Switch,
  Tag,
  Upload,
  type UploadFile,
  type UploadProps,
} from "antd";
import React, { useEffect, useMemo, useState } from "react";
import "./index.scss";

const POSITIVE_TAGS = [
  "去雾彻底",
  "色彩自然",
  "细节清晰",
  "处理速度快",
  "整体提升明显",
];
const NEGATIVE_TAGS = [
  "残留雾气",
  "色彩失真",
  "细节丢失",
  "处理速度慢",
  "无明显改善",
];

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const MAX_TAGS = 5;
const MAX_IMAGES = 3;

interface RatingCardProps {
  predLogId: number;
  algorithmName: string;
  visible: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface RatingFormValues {
  rating: number;
  comment?: string;
  isAnonymous: boolean;
}

const RatingCard: React.FC<RatingCardProps> = ({
  predLogId,
  algorithmName,
  visible,
  onClose,
  onSuccess,
}) => {
  const [form] = Form.useForm<RatingFormValues>();
  const [submitLoading, setSubmitLoading] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [rating, setRating] = useState(0);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  const availableTags = useMemo<string[]>(() => {
    if (rating === 0) return [];
    if (rating >= 4) return POSITIVE_TAGS;
    if (rating <= 2) return NEGATIVE_TAGS;
    return [...POSITIVE_TAGS, ...NEGATIVE_TAGS];
  }, [rating]);

  useEffect(() => {
    if (visible) {
      form.resetFields();
      form.setFieldsValue({ rating: 0, comment: "", isAnonymous: false });
      setRating(0);
      setTags([]);
      setFileList([]);
      setImageUrls([]);
    }
  }, [visible, form]);

  const handleTagToggle = (tag: string) => {
    setTags((prev) => {
      if (prev.includes(tag)) {
        return prev.filter((t) => t !== tag);
      }
      if (prev.length >= MAX_TAGS) {
        message.warning(`最多选择 ${MAX_TAGS} 个标签`);
        return prev;
      }
      return [...prev, tag];
    });
  };

  const handleBeforeUpload: UploadProps["beforeUpload"] = (file) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      message.error("仅支持 JPG/PNG/WEBP 格式");
      return Upload.LIST_IGNORE;
    }
    if (file.size > MAX_FILE_SIZE) {
      message.error("图片大小不能超过 5MB");
      return Upload.LIST_IGNORE;
    }
    if (fileList.length >= MAX_IMAGES) {
      message.warning(`最多上传 ${MAX_IMAGES} 张图片`);
      return Upload.LIST_IGNORE;
    }
    return true;
  };

  const handleCustomRequest: UploadProps["customRequest"] = async (options) => {
    const { file, onSuccess: onUploadSuccess, onError } = options;
    try {
      const data = await FileAPI.upload(file as File);
      setImageUrls((prev) => [...prev, data.url]);
      onUploadSuccess?.({ url: data.url }, new XMLHttpRequest());
    } catch (err: any) {
      message.error("图片上传失败：" + (err?.message || "未知错误"));
      onError?.(err);
    }
  };

  const handleRemove: UploadProps["onRemove"] = (file) => {
    const removedUrl = (file.url || file.response?.url) as string | undefined;
    setFileList((prev) => prev.filter((f) => f.uid !== file.uid));
    if (removedUrl) {
      setImageUrls((prev) => prev.filter((url) => url !== removedUrl));
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (values.rating < 1 || values.rating > 5) {
        message.error("评分范围为 1-5");
        return;
      }
      setSubmitLoading(true);
      await FeedbackAPI.createRating({
        predLogId,
        rating: values.rating,
        comment: values.comment || undefined,
        tags: tags.length ? tags : undefined,
        imageUrls: imageUrls.length ? imageUrls : undefined,
        isAnonymous: values.isAnonymous ? 1 : 0,
      });
      message.success("评价成功，获得成长值奖励");
      onSuccess?.();
      onClose();
    } catch (error: any) {
      if (error?.errorFields) return;
      message.error(error?.message || "提交失败");
    } finally {
      setSubmitLoading(false);
    }
  };

  return (
    <Modal
      title="算法效果评价"
      open={visible}
      width={600}
      confirmLoading={submitLoading}
      okText="提交评价"
      cancelText="跳过"
      destroyOnHidden
      onOk={handleSubmit}
      onCancel={onClose}
      className="rating-card-modal"
    >
      <div className="algorithm-tip">
        <PictureOutlined />
        <span>{algorithmName}</span>
      </div>

      <Form
        form={form}
        layout="horizontal"
        labelCol={{ span: 5 }}
        wrapperCol={{ span: 18 }}
        colon={false}
        validateTrigger="onBlur"
      >
        <Form.Item
          name="rating"
          label="整体评分"
          rules={[{ required: true, message: "请选择评分" }]}
        >
          <Rate
            count={5}
            value={rating}
            onChange={(val) => {
              setRating(val);
              form.setFieldValue("rating", val);
            }}
            tooltips={["很不满意", "不满意", "一般", "满意", "非常满意"]}
          />
        </Form.Item>

        {availableTags.length > 0 && (
          <Form.Item label="效果标签">
            <div className="tag-selector">
              {availableTags.map((tag) => (
                <Tag.CheckableTag
                  key={tag}
                  checked={tags.includes(tag)}
                  onChange={() => handleTagToggle(tag)}
                >
                  {tag}
                </Tag.CheckableTag>
              ))}
              <span className="tag-count-hint">
                最多选择 5 个（{tags.length}/5）
              </span>
            </div>
          </Form.Item>
        )}

        <Form.Item name="comment" label="评价内容">
          <Input.TextArea
            rows={4}
            maxLength={500}
            showCount
            placeholder="说说您的使用体验（选填）"
          />
        </Form.Item>

        <Form.Item label="上传截图">
          <Upload
            listType="picture-card"
            fileList={fileList}
            beforeUpload={handleBeforeUpload}
            customRequest={handleCustomRequest}
            onRemove={handleRemove}
            accept="image/jpeg,image/png,image/webp"
            multiple
          >
            {fileList.length < MAX_IMAGES && (
              <div>
                <PlusOutlined />
                <div className="upload-text">上传</div>
              </div>
            )}
          </Upload>
          <div className="upload-tip">
            支持 JPG/PNG/WEBP，单张不超过 5MB，最多 3 张
          </div>
        </Form.Item>

        <Form.Item name="isAnonymous" label="匿名评价" valuePropName="checked">
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default RatingCard;
