import React, { useState, useEffect, useRef, useCallback } from 'react'
import { View, Text, Image } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { ArrowLeft } from '@taroify/icons'
import CompareToolbar from '@/components/compare/CompareToolbar'
import { loadCompareContext } from '@/components/compare/types'
import './index.less'

// 放大倍数选项
const ZOOM_OPTIONS = [2, 3, 5] as const
// 放大镜尺寸选项
const SIZE_OPTIONS = [
  { value: 100, label: '小' },
  { value: 150, label: '中' },
  { value: 200, label: '大' },
] as const
// 显示模式
type DisplayMode = 'origin' | 'result' | 'compare'

const MagnifierPage: React.FC = () => {
  const [ctx, setCtx] = useState(loadCompareContext)
  const [zoom, setZoom] = useState<number>(2)
  const [lensSize, setLensSize] = useState<number>(150)
  const [displayMode, setDisplayMode] = useState<DisplayMode>('compare')
  const [lensPos, setLensPos] = useState({ x: 0, y: 0 })
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })

  const containerRef = useRef<HTMLDivElement>(null)
  const lastTapTime = useRef(0)

  useEffect(() => {
    setCtx(loadCompareContext())
  }, [])

  const { originImage, result } = ctx
  const hasResult = originImage && result?.resultUrl

  // 获取容器尺寸
  useEffect(() => {
    if (!hasResult) return
    const timer = setTimeout(() => {
      const rect = containerRef.current?.getBoundingClientRect()
      if (rect) {
        setContainerSize({ width: rect.width, height: rect.height })
        setLensPos({ x: rect.width / 2, y: rect.height / 2 })
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [hasResult])

  // 触摸移动放大镜
  const handleTouchMove = useCallback((e: any) => {
    const touch = e.touches[0]
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = touch.clientX - rect.left
    const y = touch.clientY - rect.top
    setLensPos({
      x: Math.max(0, Math.min(rect.width, x)),
      y: Math.max(0, Math.min(rect.height, y)),
    })
  }, [])

  // 双指捏合调整倍数
  const lastPinchDistance = useRef(0)
  const handleTouchStart = useCallback((e: any) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      lastPinchDistance.current = Math.sqrt(dx * dx + dy * dy)
    }
  }, [])

  const handlePinchMove = useCallback((e: any) => {
    if (e.touches.length !== 2 || lastPinchDistance.current === 0) return
    const dx = e.touches[0].clientX - e.touches[1].clientX
    const dy = e.touches[0].clientY - e.touches[1].clientY
    const distance = Math.sqrt(dx * dx + dy * dy)
    const delta = distance - lastPinchDistance.current

    if (Math.abs(delta) > 10) {
      setZoom((prev) => {
        if (delta > 0) {
          const nextIndex = Math.min(ZOOM_OPTIONS.indexOf(prev as 2 | 3 | 5) + 1, ZOOM_OPTIONS.length - 1)
          return ZOOM_OPTIONS[nextIndex]
        } else {
          const nextIndex = Math.max(ZOOM_OPTIONS.indexOf(prev as 2 | 3 | 5) - 1, 0)
          return ZOOM_OPTIONS[nextIndex]
        }
      })
      lastPinchDistance.current = distance
    }
  }, [])

  const handleTouchEnd = useCallback(() => {
    lastPinchDistance.current = 0
  }, [])

  // 点击切换显示模式（原图 → 处理后 → 对比 → 原图）
  const handleTap = useCallback(() => {
    const now = Date.now()
    if (now - lastTapTime.current < 300) {
      // 双击 - 暂不实现标记点
      return
    }
    lastTapTime.current = now
    setDisplayMode((prev) => {
      if (prev === 'origin') return 'result'
      if (prev === 'result') return 'compare'
      return 'origin'
    })
  }, [])

  // 计算放大镜内背景图位置
  const getLensBackgroundPosition = () => {
    if (!containerSize.width || !containerSize.height) return '0 0'
    const bgX = -(lensPos.x * zoom - lensSize / 2)
    const bgY = -(lensPos.y * zoom - lensSize / 2)
    return `${bgX}px ${bgY}px`
  }

  const lensStyle = (imageUrl: string): React.CSSProperties => ({
    width: `${lensSize}px`,
    height: `${lensSize}px`,
    left: `${lensPos.x - lensSize / 2}px`,
    top: `${lensPos.y - lensSize / 2}px`,
    backgroundImage: `url(${imageUrl})`,
    backgroundRepeat: 'no-repeat',
    backgroundSize: `${containerSize.width * zoom}px ${containerSize.height * zoom}px`,
    backgroundPosition: getLensBackgroundPosition(),
    borderRadius: '50%',
    border: '2px solid #fff',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
  })

  return (
    <View className="magnifier-page">
      {/* 顶部导航 */}
      <View className="navbar">
        <View className="nav-back" onClick={() => Taro.navigateBack()}>
          <ArrowLeft size="20" color="#333" />
        </View>
        <Text className="nav-title">放大镜对比</Text>
      </View>

      {/* 对比区域 */}
      {!hasResult ? (
        <View className="empty-state">
          <Text className="empty-text">暂无对比数据</Text>
          <Text className="empty-hint">请先完成去雾处理</Text>
        </View>
      ) : (
        <>
          {/* 图片容器 + 放大镜 */}
          <View
            className="image-container"
            ref={containerRef as any}
            onTouchStart={handleTouchStart}
            onTouchMove={(e: any) => {
              handleTouchMove(e)
              handlePinchMove(e)
            }}
            onTouchEnd={handleTouchEnd}
            onClick={handleTap}
          >
            <Image
              src={result!.resultUrl}
              className="base-image"
              mode="widthFix"
              lazyLoad
            />

            {/* 原图放大镜 */}
            {(displayMode === 'origin' || displayMode === 'compare') && (
              <View
                className={`magnifier-lens ${displayMode === 'compare' ? 'lens-origin' : ''}`}
                style={lensStyle(originImage!.url)}
              />
            )}

            {/* 处理后放大镜 */}
            {(displayMode === 'result' || displayMode === 'compare') && (
              <View
                className={`magnifier-lens ${displayMode === 'compare' ? 'lens-result' : ''}`}
                style={{
                  ...lensStyle(result!.resultUrl),
                  ...(displayMode === 'compare' ? { left: `${lensPos.x + lensSize / 2}px` } : {}),
                }}
              />
            )}
          </View>

          {/* 提示 */}
          <View className="magnifier-hint">
            <Text>拖动移动放大镜 · 双指捏合调整倍数 · 点击切换模式</Text>
          </View>

          {/* 控制面板 */}
          <View className="control-panel">
            {/* 显示模式 */}
            <View className="control-group">
              <Text className="control-label">显示模式</Text>
              <View className="control-options">
                {([
                  { value: 'origin', label: '原图' },
                  { value: 'result', label: '处理后' },
                  { value: 'compare', label: '对比' },
                ] as const).map((opt) => (
                  <View
                    key={opt.value}
                    className={`control-option ${displayMode === opt.value ? 'active' : ''}`}
                    onClick={() => setDisplayMode(opt.value)}
                  >
                    <Text>{opt.label}</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 放大倍数 */}
            <View className="control-group">
              <Text className="control-label">放大倍数</Text>
              <View className="control-options">
                {ZOOM_OPTIONS.map((z) => (
                  <View
                    key={z}
                    className={`control-option ${zoom === z ? 'active' : ''}`}
                    onClick={() => setZoom(z)}
                  >
                    <Text>{z}x</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 放大镜尺寸 */}
            <View className="control-group">
              <Text className="control-label">放大镜大小</Text>
              <View className="control-options">
                {SIZE_OPTIONS.map((opt) => (
                  <View
                    key={opt.value}
                    className={`control-option ${lensSize === opt.value ? 'active' : ''}`}
                    onClick={() => setLensSize(opt.value)}
                  >
                    <Text>{opt.label}</Text>
                  </View>
                ))}
              </View>
            </View>
          </View>
        </>
      )}

      {/* 底部工具栏 */}
      <CompareToolbar currentMode="magnifier" resultUrl={result?.resultUrl} />
    </View>
  )
}

export default MagnifierPage
