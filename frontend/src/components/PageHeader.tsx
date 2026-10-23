import { Typography } from "antd";
import type { ReactNode } from "react";

interface Props {
  title: ReactNode;
  subtitle?: ReactNode;
  extra?: ReactNode;
  before?: ReactNode;
}

export default function PageHeader({ title, subtitle, extra, before }: Props) {
  return (
    <div className="page-header">
      <div style={{ display: "flex", gap: 12, alignItems: "center", minWidth: 0 }}>
        {before}
        <div style={{ minWidth: 0 }}>
          <Typography.Title level={1} className="page-title">
            {title}
          </Typography.Title>
          {subtitle && <div className="page-subtitle">{subtitle}</div>}
        </div>
      </div>
      {extra && <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{extra}</div>}
    </div>
  );
}
