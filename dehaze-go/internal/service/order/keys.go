package order

import "fmt"

// 订单模块缓存键与分布式锁键集中管理
// 定时任务调度锁（job:order:*）已移至 pkg/job/locks.go，由 JobHandler 持有
const (
	orderDetailKeyFmt  = "order:detail:%s"
	orderCreateLockFmt = "order:create:lock:%d:%d"
)

func OrderDetailKey(orderNo string) string { return fmt.Sprintf(orderDetailKeyFmt, orderNo) }
func OrderCreateLockKey(userID, packageID int64) string {
	return fmt.Sprintf(orderCreateLockFmt, userID, packageID)
}
