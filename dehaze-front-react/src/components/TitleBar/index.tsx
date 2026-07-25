import defaultSettings from "@/settings";
import React, { useState } from "react";

import "./index.scss";

export default function TitleBar() {
  const [isMaximized, setIsMaximized] = useState(false);

  const handleMinimize = () => window.electronAPI?.minimize();
  const handleToggleMaximize = () => {
    window.electronAPI?.toggleMaximize();
    setIsMaximized((v) => !v);
  };
  const handleClose = () => window.electronAPI?.close();

  return (
    <div className="titlebar">
      <div className="titlebar__left">
        <img src="/favicon.ico" className="titlebar__logo" alt="logo" />
        <span className="titlebar__title">{defaultSettings.title}</span>
      </div>
      <div className="titlebar__controls">
        <button
          className="titlebar__btn"
          title="最小化"
          onClick={handleMinimize}
        >
          <svg width="10" height="10" viewBox="0 0 10 10">
            <path d="M0 5h10" stroke="currentColor" strokeWidth="1" />
          </svg>
        </button>
        <button
          className="titlebar__btn"
          title="最大化"
          onClick={handleToggleMaximize}
        >
          {isMaximized ? (
            <svg width="10" height="10" viewBox="0 0 10 10">
              <rect
                x="0.5"
                y="2.5"
                width="7"
                height="7"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
              <rect
                x="2.5"
                y="0.5"
                width="7"
                height="7"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 10 10">
              <rect
                x="0.5"
                y="0.5"
                width="9"
                height="9"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
            </svg>
          )}
        </button>
        <button
          className="titlebar__btn titlebar__btn--close"
          title="关闭"
          onClick={handleClose}
        >
          <svg width="10" height="10" viewBox="0 0 10 10">
            <path
              d="M0 0l10 10M10 0L0 10"
              stroke="currentColor"
              strokeWidth="1"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
