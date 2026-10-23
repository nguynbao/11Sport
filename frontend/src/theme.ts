import type { ThemeConfig } from "antd";

/**
 * Design tokens — generated with ui-ux-pro-max (--design-system "AI video dubbing internal tool dashboard").
 * Style: Soft UI Evolution · Palette: calm indigo + success emerald · Type: Be Vietnam Pro ("Vietnamese Friendly").
 * Primary darkened #6366F1 -> #4F46E5 so white text on primary passes WCAG 4.5:1.
 */
export const tokens = {
  primary: "#4F46E5",
  primaryHover: "#4338CA",
  secondary: "#818CF8",
  accent: "#059669",
  background: "#F5F3FF",
  foreground: "#1E1B4B",
  card: "#FFFFFF",
  muted: "#EEF0FA",
  mutedForeground: "#475569",
  border: "#E0E7FF",
  destructive: "#DC2626",
  warning: "#B45309",
  ring: "#4F46E5",
  fontFamily:
    "'Be Vietnam Pro', 'Noto Sans', system-ui, -apple-system, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif",
};

export const theme: ThemeConfig = {
  cssVar: true,
  token: {
    colorPrimary: tokens.primary,
    colorInfo: tokens.primary,
    colorSuccess: tokens.accent,
    colorWarning: tokens.warning,
    colorError: tokens.destructive,
    colorBgLayout: tokens.background,
    colorBgContainer: tokens.card,
    colorText: tokens.foreground,
    colorTextSecondary: tokens.mutedForeground,
    colorTextTertiary: "#64748B",
    colorBorder: "#D9DEF5",
    colorBorderSecondary: tokens.border,
    colorLink: tokens.primary,
    fontFamily: tokens.fontFamily,
    fontSize: 14,
    borderRadius: 10,
    borderRadiusLG: 14,
    controlHeight: 38,
    controlHeightLG: 44,
    motionDurationMid: "0.2s",
    boxShadow: "0 1px 2px rgba(30, 27, 75, 0.06), 0 4px 16px rgba(79, 70, 229, 0.06)",
    boxShadowSecondary: "0 8px 28px rgba(30, 27, 75, 0.12)",
  },
  components: {
    Layout: { siderBg: "#FFFFFF", headerBg: "transparent", bodyBg: tokens.background },
    Menu: {
      itemBg: "transparent",
      itemSelectedBg: "#EEF2FF",
      itemSelectedColor: tokens.primary,
      itemHoverBg: "#F5F7FF",
      itemBorderRadius: 10,
      itemHeight: 44,
      iconSize: 18,
    },
    Card: { headerFontSize: 16, paddingLG: 22 },
    Table: { headerBg: "#F8F7FF", headerColor: tokens.mutedForeground, rowHoverBg: "#FAFAFF", cellPaddingBlock: 14 },
    Button: { fontWeight: 500, primaryShadow: "0 2px 8px rgba(79, 70, 229, 0.25)" },
    Steps: { iconSize: 30 },
    Tag: { defaultBg: tokens.muted },
  },
};
