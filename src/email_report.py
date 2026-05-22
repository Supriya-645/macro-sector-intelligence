"""
Email Summary Report — Phase Enhancement 4.

Builds an HTML daily macro intelligence summary and sends it via SMTP.
Designed to be scheduled daily by APScheduler in the FastAPI server.
"""

import logging
import os
import smtplib
import base64
import tempfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "macro_matplotlib"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.utils import get_data_paths

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inline Jinja2-style HTML email template
EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Macro Intelligence Daily Report</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background-color: #0d1117;
      color: #c9d1d9;
      margin: 0; padding: 0;
    }}
    .container {{
      max-width: 700px;
      margin: 30px auto;
      background: #161b22;
      border-radius: 12px;
      border: 1px solid #30363d;
      overflow: hidden;
    }}
    .header {{
      background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
      padding: 28px 36px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.5px;
    }}
    .header p {{
      margin: 6px 0 0;
      font-size: 13px;
      color: rgba(255,255,255,0.75);
    }}
    .section {{
      padding: 24px 36px;
      border-bottom: 1px solid #21262d;
    }}
    .section h2 {{
      font-size: 15px;
      font-weight: 600;
      color: #58a6ff;
      margin: 0 0 16px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .metric-grid {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .metric-box {{
      flex: 1;
      min-width: 130px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-box .label {{
      font-size: 11px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 6px;
    }}
    .metric-box .value {{
      font-size: 20px;
      font-weight: 700;
      color: #f0f6fc;
    }}
    .metric-box .regime-badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      background: #21262d;
      color: #8b949e;
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid #21262d;
      color: #c9d1d9;
    }}
    .positive {{ color: #3fb950; font-weight: 600; }}
    .negative {{ color: #f85149; font-weight: 600; }}
    .footer {{
      padding: 20px 36px;
      font-size: 11px;
      color: #484f58;
      text-align: center;
      line-height: 1.6;
    }}
    .chart {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #30363d;
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>&#9685; Macro Intelligence</h1>
      <p>Daily Summary Report &mdash; {date}</p>
    </div>

    <div class="section">
      <h2>Market Snapshot</h2>
      <div class="metric-grid">
        <div class="metric-box">
          <div class="label">Economic Regime</div>
          <div class="value">
            <span class="regime-badge" style="background:{regime_color}20; color:{regime_color};">
              {regime}
            </span>
          </div>
        </div>
        <div class="metric-box">
          <div class="label">Nifty 50</div>
          <div class="value">{nifty_50}</div>
        </div>
        <div class="metric-box">
          <div class="label">India VIX</div>
          <div class="value">{india_vix}</div>
        </div>
        <div class="metric-box">
          <div class="label">Fed Funds Rate</div>
          <div class="value">{fed_rate}%</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Nifty 50 Trend</h2>
      {trend_chart}
    </div>

    <div class="section">
      <h2>Top Sectors by Risk-Adjusted Return ({regime})</h2>
      <table>
        <thead>
          <tr>
            <th>Sector</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown</th>
            <th>VaR 95%</th>
          </tr>
        </thead>
        <tbody>
          {sector_rows}
        </tbody>
      </table>
    </div>

    <div class="footer">
      <p>This report is auto-generated by the Macro Intelligence Platform and is for informational
      purposes only. It does not constitute investment advice. Past performance is not indicative
      of future results. Always consult a qualified financial professional before making investment
      decisions.</p>
      <p>Generated at {datetime} UTC &mdash; Macro AI v2.1</p>
    </div>
  </div>
</body>
</html>
"""

REGIME_COLORS = {
    "Expansion": "#3fb950",
    "Peak": "#d29922",
    "Contraction": "#f85149",
    "Recovery": "#58a6ff",
}


def _build_trend_chart(master: pd.DataFrame) -> str:
    """Return an embedded base64 chart for the latest Nifty 50 trend."""
    if "Nifty_50" not in master.columns:
        return "<p>Nifty 50 trend data unavailable.</p>"

    chart_df = master[["Nifty_50"]].dropna().tail(24)
    if chart_df.empty:
        return "<p>Nifty 50 trend data unavailable.</p>"

    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=140)
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    ax.plot(chart_df.index, chart_df["Nifty_50"], color="#58a6ff", linewidth=2)
    ax.fill_between(chart_df.index, chart_df["Nifty_50"], color="#58a6ff", alpha=0.15)
    ax.set_title("Last 24 Months", color="#c9d1d9", fontsize=10, pad=10)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(color="#30363d", alpha=0.35)
    fig.autofmt_xdate(rotation=25)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="chart" alt="Nifty 50 24-month trend" src="data:image/png;base64,{encoded}" />'


def build_report_html() -> str:
    """Build an HTML email summarising the latest macro intelligence data.

    Loads the master dataset, regime labels, and risk metrics to
    extract key figures: current regime, latest Nifty 50 / India VIX /
    Fed Funds Rate, and the top 5 sectors by Sharpe ratio in the
    current regime.

    Returns:
        A complete HTML string ready to be embedded in an email body.
    """
    paths = get_data_paths()
    today = datetime.utcnow()
    date_str = today.strftime("%A, %d %B %Y")
    dt_str = today.strftime("%Y-%m-%d %H:%M")

    # Load master dataset
    try:
        master = pd.read_csv(
            paths["processed"] / "master_dataset.csv",
            index_col="Date",
            parse_dates=True,
        )
        latest = master.iloc[-1]
        nifty_50 = f"{latest.get('Nifty_50', 0):,.0f}"
        india_vix = f"{latest.get('India_VIX', 0):.2f}"
        fed_rate = f"{latest.get('US_Fed_Funds_Rate', 0):.2f}"
        trend_chart = _build_trend_chart(master)
    except Exception:
        nifty_50 = "N/A"
        india_vix = "N/A"
        fed_rate = "N/A"
        trend_chart = "<p>Trend chart unavailable.</p>"

    # Load current regime
    regime = "Unknown"
    try:
        regime_df = pd.read_csv(
            paths["processed"] / "regime_labels.csv",
            index_col="Date",
            parse_dates=True,
        )
        regime = regime_df["Regime"].iloc[-1]
    except Exception:
        pass

    regime_color = REGIME_COLORS.get(regime, "#8b949e")

    # Load risk metrics for current regime
    sector_rows = ""
    try:
        risk_df = pd.read_csv(paths["processed"] / "risk_metrics.csv")
        regime_risk = risk_df[risk_df["Regime"].str.lower() == regime.lower()]
        if regime_risk.empty:
            regime_risk = risk_df[risk_df["Regime"] == "All Regimes"]

        top5 = regime_risk.sort_values("Sharpe_Ratio", ascending=False).head(5)

        for _, row in top5.iterrows():
            sharpe = f"{row['Sharpe_Ratio']:.3f}" if pd.notna(row.get("Sharpe_Ratio")) else "N/A"
            dd = f"{row['Max_Drawdown'] * 100:.2f}%" if pd.notna(row.get("Max_Drawdown")) else "N/A"
            var95 = f"{row['VaR_95'] * 100:.2f}%" if pd.notna(row.get("VaR_95")) else "N/A"
            sector_name = str(row.get("Sector", "")).replace("_", " ")
            dd_cls = "negative" if row.get("Max_Drawdown", 0) < 0 else ""
            sector_rows += (
                f"<tr>"
                f"<td><strong>{sector_name}</strong></td>"
                f"<td class='positive'>{sharpe}</td>"
                f"<td class='{dd_cls}'>{dd}</td>"
                f"<td class='negative'>{var95}</td>"
                f"</tr>\n"
            )
    except Exception as exc:
        sector_rows = f"<tr><td colspan='4'>Risk data unavailable: {exc}</td></tr>"

    return EMAIL_TEMPLATE.format(
        date=date_str,
        datetime=dt_str,
        regime=regime,
        regime_color=regime_color,
        nifty_50=nifty_50,
        india_vix=india_vix,
        fed_rate=fed_rate,
        trend_chart=trend_chart,
        sector_rows=sector_rows,
    )


def send_email_report(recipients: Optional[list] = None) -> bool:
    """Send the daily macro intelligence HTML email report.

    Builds the HTML content and transmits it via SMTP using credentials
    stored in environment variables. Supports both SMTP_SSL (port 465)
    and STARTTLS (port 587).

    Args:
        recipients: List of recipient email addresses. If empty or None,
            falls back to the ``REPORT_RECIPIENTS`` environment variable
            (comma-separated).

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    env_recipients = os.getenv("REPORT_RECIPIENTS") or os.getenv("EMAIL_RECIPIENTS", "")

    if not recipients:
        recipients = [r.strip() for r in env_recipients.split(",") if r.strip()]

    if not recipients:
        logger.error("No recipients specified. Set REPORT_RECIPIENTS or EMAIL_RECIPIENTS in .env.")
        return False

    placeholder_values = {"your_email@gmail.com", "your_app_password", "your_password"}
    if not smtp_user or not smtp_pass or smtp_user in placeholder_values or smtp_pass in placeholder_values:
        logger.error("SMTP credentials not configured. Set SMTP_USER and SMTP_PASS in .env.")
        return False

    try:
        html_body = build_report_html()
        today = datetime.utcnow().strftime("%Y-%m-%d")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Macro Intelligence — Daily Report — {today}"
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())

        logger.info("Daily report sent to: %s", recipients)
        return True

    except Exception as exc:
        logger.error("Failed to send email report: %s", exc)
        return False


if __name__ == "__main__":
    print("=== Building email report HTML (no SMTP send) ===\n")
    html = build_report_html()
    output_path = Path("data/processed/daily_report_preview.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path}")
    print(f"Length: {len(html)} chars")
