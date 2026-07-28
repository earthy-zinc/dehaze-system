import { OrderAPI, type PayMethod, type PayResult } from "dehaze-sdk-js";
import { Button, Modal, Radio, Result, message } from "antd";
import QRCode from "qrcode";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";

const PAY_METHOD_OPTIONS: { label: string; value: PayMethod }[] = [
  { label: "微信支付", value: "wechat" },
  { label: "支付宝", value: "alipay" },
  { label: "余额支付", value: "balance" },
];

export interface PaymentDialogRef {
  open: (orderNo: string, payableAmount: number) => void;
}

interface PaymentDialogProps {
  onPaid?: () => void;
}

const PaymentDialog = forwardRef<PaymentDialogRef, PaymentDialogProps>(
  ({ onPaid }, ref) => {
    const [visible, setVisible] = useState(false);
    const [loading, setLoading] = useState(false);
    const [orderNo, setOrderNo] = useState("");
    const [payableAmount, setPayableAmount] = useState(0);
    const [payMethod, setPayMethod] = useState<PayMethod>("wechat");
    const [payResult, setPayResult] = useState<PayResult | null>(null);
    const [qrCodeDataUrl, setQrCodeDataUrl] = useState("");

    const open = useCallback((no: string, amount: number) => {
      setOrderNo(no);
      setPayableAmount(amount);
      setPayMethod("wechat");
      setPayResult(null);
      setQrCodeDataUrl("");
      setVisible(true);
    }, []);

    useImperativeHandle(ref, () => ({ open }), [open]);

    const handleClose = useCallback(() => {
      setVisible(false);
      setPayResult(null);
      setQrCodeDataUrl("");
    }, []);

    const handlePay = useCallback(async () => {
      setLoading(true);
      setQrCodeDataUrl("");
      try {
        const result = await OrderAPI.pay(orderNo, { payMethod });
        setPayResult(result);
        if (result.paid) {
          message.success("支付成功");
        } else {
          const qrContent = result.qrCode || result.payUrl || "";
          if (qrContent) {
            const url = await QRCode.toDataURL(qrContent, {
              width: 240,
              margin: 2,
              color: { dark: "#000000", light: "#ffffff" },
            });
            setQrCodeDataUrl(url);
          }
        }
      } catch (error: any) {
        message.error(error?.message || "支付失败");
      } finally {
        setLoading(false);
      }
    }, [orderNo, payMethod]);

    const handleComplete = useCallback(() => {
      handleClose();
      onPaid?.();
    }, [handleClose, onPaid]);

    const payHint =
      payResult?.payMethod === "wechat"
        ? "请使用微信扫码完成支付"
        : payResult?.payMethod === "alipay"
          ? "请使用支付宝扫码完成支付"
          : "请完成支付";

    return (
      <Modal
        title="订单支付"
        open={visible}
        width={420}
        destroyOnHidden
        onCancel={handleClose}
        footer={
          payResult ? (
            <>
              <Button onClick={handleClose}>关闭</Button>
              {payResult.paid ? (
                <Button type="primary" onClick={handleComplete}>
                  完成
                </Button>
              ) : (
                <Button type="primary" loading={loading} onClick={handlePay}>
                  重新支付
                </Button>
              )}
            </>
          ) : (
            <>
              <Button type="primary" loading={loading} onClick={handlePay}>
                确认支付
              </Button>
              <Button onClick={handleClose}>取消</Button>
            </>
          )
        }
      >
        <div className="payment-summary">
          <div className="payment-row">
            <span className="label">订单号：</span>
            <span>{orderNo}</span>
          </div>
          <div className="payment-row">
            <span className="label">应付金额：</span>
            <span className="pay-amount">¥{payableAmount.toFixed(2)}</span>
          </div>
        </div>

        {!payResult && (
          <div style={{ marginTop: 16 }}>
            <div style={{ marginBottom: 8 }}>支付方式</div>
            <Radio.Group
              value={payMethod}
              onChange={(e) => setPayMethod(e.target.value)}
              buttonStyle="solid"
            >
              {PAY_METHOD_OPTIONS.map((opt) => (
                <Radio.Button key={opt.value} value={opt.value}>
                  {opt.label}
                </Radio.Button>
              ))}
            </Radio.Group>
          </div>
        )}

        {payResult && (
          <div className="pay-result-box">
            {payResult.paid ? (
              <Result
                status="success"
                title="支付成功"
                subTitle="订单已支付完成"
              />
            ) : (
              <>
                <Result status="info" title="请扫码支付" subTitle={payHint} />
                {qrCodeDataUrl && (
                  <div className="pay-qr-code">
                    <img
                      src={qrCodeDataUrl}
                      alt="支付二维码"
                      className="qr-img"
                    />
                  </div>
                )}
                {payResult.payUrl && (
                  <div className="pay-link-row">
                    <a
                      href={payResult.payUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      点击打开支付页面 →
                    </a>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </Modal>
    );
  }
);

PaymentDialog.displayName = "PaymentDialog";

export default PaymentDialog;
