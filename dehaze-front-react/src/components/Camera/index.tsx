import { Button, message, Select } from "antd";
import React, { createRef, useEffect, useMemo, useState } from "react";
import style from "./index.module.scss";

export interface CameraProps {
  onSave: (file: File) => void;
  onCancel: () => void;
}

export default function Camera({ onSave, onCancel }: CameraProps) {
  const videoRef = createRef<HTMLVideoElement>();
  const canvasRef = createRef<HTMLCanvasElement>();
  const [image, setImage] = useState<File>();
  const [stream, setStream] = useState<MediaStream>();
  const [videoInfo, setVideoInfo] = useState({
    selectedDevice: {} as MediaDeviceInfo,
    device: [] as MediaDeviceInfo[],
    width: 0,
    height: 0,
    videoWidth: 0,
    videoHeight: 0,
  });
  const isTakenPhoto = useMemo(() => image !== undefined, [image]);

  function startCamera(deviceId?: string) {
    if (!videoRef.current) return;
    navigator.mediaDevices
      .getUserMedia({
        video: {
          deviceId: deviceId ? deviceId : videoInfo.selectedDevice.deviceId,
        },
      })
      .then((stream) => {
        setStream(stream);
        videoRef.current!.srcObject = stream;
      })
      .then(() => {
        videoRef.current!.onloadedmetadata = () => {
          const videoRect = videoRef.current!.getBoundingClientRect();
          setVideoInfo({
            ...videoInfo,
            width: videoRect.width,
            height: videoRect.height,
            videoWidth: videoRef.current!.videoWidth,
            videoHeight: videoRef.current!.videoHeight,
          });
        };
      })
      .catch((err) => {
        message.error("无法获取摄像头信息，请检查设备", err.message);
      });
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  }

  function capturePhoto() {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current!;
    canvas.width = videoInfo.videoWidth;
    canvas.height = videoInfo.videoHeight;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(videoRef.current!, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        setImage(new File([blob], "photo.png", { type: blob.type }));
      }
    });
  }

  function retakePhoto() {
    setImage(undefined);
    if (!canvasRef.current) return;
    canvasRef.current.style.display = "none";
  }

  function savePhoto() {
    if (!image) return;
    onSave(image);
  }

  function cancelTake() {
    onCancel();
    stopCamera();
  }

  function handleSelectChange(device: MediaDeviceInfo) {
    startCamera(device.deviceId);
  }

  useEffect(() => {
    navigator.mediaDevices
      .enumerateDevices()
      .then((res) => {
        const device = res.filter((device) => device.kind === "videoinput");
        setVideoInfo({
          ...videoInfo,
          device: device,
          selectedDevice: device[0],
        });
      })
      .then(() => startCamera())
      .catch((err: any) => {
        message.error("无法获取摄像头信息，请检查设备", err.message);
      });
    return () => stopCamera();
  }, []);

  return (
    <div>
      <div className={style["video-wrapper"]}>
        <video ref={videoRef} autoPlay={true}></video>
        <canvas ref={canvasRef}></canvas>
      </div>

      <div className={style["controller"]}>
        <span className={"mr-2"} style={{ lineHeight: "32px" }}>
          切换摄像头
        </span>
        <Select
          disabled={isTakenPhoto}
          className={"mr-6"}
          placeholder={"请选择摄像头"}
          options={videoInfo.device.map((device) => ({
            label: device.label,
            value: device,
          }))}
          value={videoInfo.selectedDevice}
          onChange={handleSelectChange}
        />
        {!isTakenPhoto && (
          <Button type="primary" className={"ml-4"} onClick={capturePhoto}>
            拍照
          </Button>
        )}
        {isTakenPhoto && (
          <>
            <Button onClick={retakePhoto}>重拍</Button>
            <Button type="primary" onClick={savePhoto}>
              保存
            </Button>
          </>
        )}
        <Button onClick={cancelTake}>取消</Button>
      </div>
    </div>
  );
}
