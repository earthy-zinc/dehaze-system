package member

import "fmt"

// 会员模块缓存键集中管理，避免字面量散落各处
// 定时任务调度锁（job:member:*）已移至 pkg/job/locks.go，由 JobHandler 持有
const (
	memberProfileKeyFmt = "member:profile:%d"
	memberLevelKeyFmt   = "member:level:%d"
	memberBenefitKeyFmt = "member:benefit:%s"
	memberBenefitAllKey = "member:benefit:all"
	memberQuotaKeyFmt   = "member:quota:%d:%s"
)

func MemberProfileKey(userID int64) string  { return fmt.Sprintf(memberProfileKeyFmt, userID) }
func MemberLevelKey(userID int64) string    { return fmt.Sprintf(memberLevelKeyFmt, userID) }
func MemberBenefitKey(levelCode string) string { return fmt.Sprintf(memberBenefitKeyFmt, levelCode) }
func MemberBenefitAllKey() string           { return memberBenefitAllKey }
func MemberQuotaKey(userID int64, quotaType QuotaType) string {
	return fmt.Sprintf(memberQuotaKeyFmt, userID, quotaType)
}
