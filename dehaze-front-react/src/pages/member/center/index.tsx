import {
  MemberAPI,
  type MemberLevelCode,
  type MemberProfileVO,
  type SignInCalendarVO,
} from "dehaze-sdk-js";
import {
  Button,
  Calendar,
  Card,
  Col,
  Progress,
  Row,
  Spin,
  Tag,
  message,
} from "antd";
import {
  ArrowUpOutlined,
  CalendarOutlined,
  StarOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.scss";

const LEVEL_GRADIENT_MAP: Record<MemberLevelCode, string> = {
  level_0: "linear-gradient(135deg, #8c8c8c 0%, #595959 100%)",
  level_1: "linear-gradient(135deg, #409eff 0%, #1677ff 100%)",
  level_2: "linear-gradient(135deg, #722ed1 0%, #531dab 100%)",
  level_3: "linear-gradient(135deg, #fa8c16 0%, #d4380d 100%)",
};

function formatToday(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate()
  ).padStart(2, "0")}`;
}

const MemberCenter: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState<MemberProfileVO | undefined>(
    undefined
  );
  const [calendar, setCalendar] = useState<SignInCalendarVO | undefined>(
    undefined
  );
  const [calendarYear, setCalendarYear] = useState(() =>
    new Date().getFullYear()
  );
  const [calendarMonth, setCalendarMonth] = useState(
    () => new Date().getMonth() + 1
  );
  const [signInLoading, setSignInLoading] = useState(false);
  const [bonusVisible, setBonusVisible] = useState(false);
  const [bonusGrowth, setBonusGrowth] = useState(0);

  const todayStr = useMemo(() => formatToday(), []);

  const loadProfile = useCallback(() => {
    setLoading(true);
    MemberAPI.getProfile()
      .then((data) => {
        setProfile(data);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const loadCalendar = useCallback((year: number, month: number) => {
    MemberAPI.getSignInCalendar(year, month).then((data) => {
      setCalendar(data);
    });
  }, []);

  useEffect(() => {
    loadProfile();
    loadCalendar(calendarYear, calendarMonth);
  }, [loadProfile, loadCalendar, calendarYear, calendarMonth]);

  const isMaxLevel = profile?.levelCode === "level_3";
  const growthToNext =
    profile?.nextLevelGrowth != null
      ? Math.max(0, profile.nextLevelGrowth - profile.growthValue)
      : 0;
  const dehazeRemaining = profile
    ? Math.max(0, profile.monthlyDehazeQuota - profile.monthlyDehazeUsed)
    : 0;
  const evaluateRemaining = profile
    ? Math.max(0, profile.monthlyEvaluateQuota - profile.monthlyEvaluateUsed)
    : 0;

  const hasAnyUnlockedFeature = profile?.benefits
    ? !!(
        profile.benefits.hdExport ||
        profile.benefits.reportExport ||
        profile.benefits.batchDownload ||
        profile.benefits.advancedParams
      )
    : false;

  const hasSignedToday = calendar?.signDates?.includes(todayStr) ?? false;

  const isSignedDate = useCallback(
    (dayStr: string) => {
      return calendar?.signDates?.includes(dayStr) ?? false;
    },
    [calendar]
  );

  const handleSignIn = useCallback(() => {
    setSignInLoading(true);
    MemberAPI.signIn()
      .then((res) => {
        const total = res.growthValue + res.bonusGrowth;
        setBonusGrowth(total);
        setBonusVisible(true);
        setTimeout(() => setBonusVisible(false), 2000);
        message.success(
          `签到成功！连续签到 ${res.continuousDays} 天，获得 ${total} 成长值`
        );
        loadProfile();
        loadCalendar(calendarYear, calendarMonth);
      })
      .finally(() => {
        setSignInLoading(false);
      });
  }, [loadProfile, loadCalendar, calendarYear, calendarMonth]);

  const handleUpgrade = useCallback(() => {
    message.info("升级功能即将开放，敬请期待");
  }, []);

  return (
    <div className="member-center">
      <div
        className="level-card"
        style={{
          background: profile
            ? LEVEL_GRADIENT_MAP[profile.levelCode]
            : LEVEL_GRADIENT_MAP.level_0,
        }}
      >
        <Spin spinning={loading}>
          <div className="level-info">
            <div className="level-icon">
              <TrophyOutlined style={{ fontSize: 56, color: "#fff" }} />
            </div>
            <div className="level-detail">
              <div className="level-name">
                {profile?.levelName || "未开通会员"}
              </div>
              <div className="level-meta">
                {profile && (
                  <>
                    {profile.expireTime ? (
                      <span className="meta-item">
                        <CalendarOutlined />
                        到期时间：{profile.expireTime}
                      </span>
                    ) : (
                      <span className="meta-item growth-maintain">
                        <StarOutlined />
                        成长值维持
                      </span>
                    )}
                    <span className="meta-item">
                      <ArrowUpOutlined />
                      成长值：{profile.growthValue}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
        </Spin>
        <div className="level-actions">
          <Button ghost onClick={() => navigate("/member/growth-logs")}>
            成长值明细
          </Button>
          {!isMaxLevel && profile && (
            <Button type="primary" onClick={handleUpgrade}>
              升级
            </Button>
          )}
        </div>
      </div>

      {profile && (
        <Card className="growth-card" size="small">
          <div className="growth-header">
            <span className="growth-title">成长值进度</span>
            <span className="growth-current">{profile.growthValue}</span>
          </div>
          {!isMaxLevel ? (
            <>
              <Progress
                percent={profile.progressPercent}
                strokeWidth={14}
                strokeColor={{
                  "0%": "#409eff",
                  "100%": "#722ed1",
                }}
              />
              <div className="growth-footer">
                距下一等级还需
                <strong className="growth-gap">{growthToNext}</strong>
                成长值
              </div>
            </>
          ) : (
            <div className="max-level-text">
              <TrophyOutlined />
              已达最高等级
            </div>
          )}
        </Card>
      )}

      {profile && (
        <Card className="benefit-card" size="small" title="权益概览（本月）">
          <Row gutter={20}>
            <Col sm={6} xs={12}>
              <div className="benefit-item">
                <div className="benefit-label">本月去雾剩余</div>
                <div className="benefit-value">{dehazeRemaining}</div>
                <div className="benefit-sub">
                  总额 {profile.monthlyDehazeQuota} / 已用{" "}
                  {profile.monthlyDehazeUsed}
                </div>
              </div>
            </Col>
            <Col sm={6} xs={12}>
              <div className="benefit-item">
                <div className="benefit-label">本月评估剩余</div>
                <div className="benefit-value">{evaluateRemaining}</div>
                <div className="benefit-sub">
                  总额 {profile.monthlyEvaluateQuota} / 已用{" "}
                  {profile.monthlyEvaluateUsed}
                </div>
              </div>
            </Col>
            <Col sm={6} xs={12}>
              <div className="benefit-item">
                <div className="benefit-label">批量上限</div>
                <div className="benefit-value">
                  {profile.benefits?.batchLimit ?? 0}
                </div>
                <div className="benefit-sub">单次批量处理数量</div>
              </div>
            </Col>
            <Col sm={6} xs={12}>
              <div className="benefit-item">
                <div className="benefit-label">历史保留</div>
                <div className="benefit-value">
                  {profile.benefits?.historyRetention ?? 0}
                </div>
                <div className="benefit-sub">历史记录保留天数</div>
              </div>
            </Col>
          </Row>

          <div className="unlocked-features">
            <div className="features-title">已解锁功能</div>
            <div className="features-list">
              {profile.benefits?.hdExport ? (
                <Tag color="success" icon={<StarOutlined />}>
                  高清导出
                </Tag>
              ) : null}
              {profile.benefits?.reportExport ? (
                <Tag color="success" icon={<StarOutlined />}>
                  报告导出
                </Tag>
              ) : null}
              {profile.benefits?.batchDownload ? (
                <Tag color="success" icon={<StarOutlined />}>
                  批量下载
                </Tag>
              ) : null}
              {profile.benefits?.advancedParams ? (
                <Tag color="success" icon={<StarOutlined />}>
                  高级参数
                </Tag>
              ) : null}
              {!hasAnyUnlockedFeature && (
                <Tag color="default">暂无解锁功能</Tag>
              )}
            </div>
          </div>
        </Card>
      )}

      <Card
        className="signin-card"
        size="small"
        title="每日签到"
        extra={
          <Button
            type="primary"
            loading={signInLoading}
            disabled={hasSignedToday}
            icon={<CalendarOutlined />}
            onClick={handleSignIn}
          >
            {hasSignedToday ? "今日已签到" : "立即签到"}
          </Button>
        }
      >
        <div className="signin-stats">
          <div className="stat-item">
            <span className="stat-label">连续签到</span>
            <span className="stat-value">
              {calendar?.continuousDays ?? 0} 天
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">累计签到</span>
            <span className="stat-value">{calendar?.totalDays ?? 0} 天</span>
          </div>
        </div>

        <Calendar
          fullscreen={false}
          onPanelChange={(date) => {
            setCalendarYear(date.year());
            setCalendarMonth(date.month() + 1);
          }}
          cellRender={(date) => {
            const dayStr = date.format("YYYY-MM-DD");
            const signed = isSignedDate(dayStr);
            const isToday = dayStr === todayStr;
            return (
              <div
                className={`calendar-cell ${signed ? "signed" : ""} ${
                  isToday ? "today" : ""
                }`}
              >
                <span className="cell-day">{date.date()}</span>
                {signed && (
                  <StarOutlined
                    className="signed-icon"
                    style={{ fontSize: 12 }}
                  />
                )}
              </div>
            );
          }}
        />
      </Card>

      {bonusVisible && (
        <div className="bonus-tip">
          <StarOutlined />
          <span>+{bonusGrowth} 成长值</span>
        </div>
      )}
    </div>
  );
};

export default MemberCenter;
