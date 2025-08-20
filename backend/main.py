# backend/main.py
# DataWise Backend
# A FastAPI server that:
# - Serves a web frontend
# - Accepts file uploads
# - Generates auto EDA reports (HTML only)
# - Embeds Top Insights directly into the report
# - Allows HTML report download
# - Works for ANY user: students, businesses, researchers, analysts

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import pandas as pd
from ydata_profiling import ProfileReport

# ===========================
# 🚀 Initialize the FastAPI App
# ===========================
app = FastAPI(
    title="DataWise",
    description="An open-source tool to get wise insights from your data — no coding needed.",
    version="0.1.0"
)

# ==================================
# 📁 Setup Upload Folder
# ==================================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==================================
# 🌐 Serve the Frontend (Web Page)
# ==================================
@app.get("/")
async def serve_frontend():
    """
    Serves the main HTML page.
    This is the user's starting point.
    """
    return FileResponse("frontend/index.html")


# ==================================
# 🔍 Health Check Endpoint
# ==================================
@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


# ==================================
# 📤 File Upload & Analysis Endpoint
# ==================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads file, reads it, auto-detects column meanings, generates EDA report.
    Works with any dataset — no strict column names required.
    Returns a summary and download links.
    """
    try:
        # Validate file type
        if not file.filename.endswith((".csv", ".xlsx", ".xls")):
            return JSONResponse(
                status_code=400,
                content={"error": "File must be a CSV or Excel file (.csv, .xls, .xlsx)"}
            )

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(1024):
                    buffer.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

        # Read into DataFrame
        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

        # --- Auto-Detect Columns by Keywords ---
        def find_column_with_keywords(df, keywords):
            """Find a column that matches any keyword (case-insensitive)"""
            for col in df.columns:
                col_lower = col.lower()
                for kw in keywords:
                    if kw.lower() in col_lower:
                        return col
            return None

        # Define keyword groups for flexible detection
        COLUMNS = {
            "revenue": ["revenue", "sales", "total", "amount", "income", "value"],
            "units_sold": ["units sold", "quantity", "qty", "count", "volume"],
            "price": ["price", "cost", "rate", "unit price"],
            "product": ["product", "item", "supplement", "name", "title"],
            "category": ["category", "type", "kind", "class"],
            "region": ["region", "location", "country", "city", "market", "area"],
            "platform": ["platform", "source", "channel", "website", "store"],
            "date": ["date", "order date", "week", "month", "timestamp", "time"]
        }

        # Detect columns
        detected = {}
        for role, keywords in COLUMNS.items():
            detected[role] = find_column_with_keywords(df, keywords)

        # --- Generate Top Insights (Context-Aware) ---
        insights = []

        # Total Revenue
        if detected["revenue"]:
            try:
                revenue_data = pd.to_numeric(df[detected["revenue"]], errors='coerce')
                total_revenue = revenue_data.sum()
                if pd.notna(total_revenue):
                    insights.append(f"💰 **Total Revenue**: ${total_revenue:,.2f}")
                else:
                    insights.append("⚠️ **Revenue Data**: Could not calculate total revenue (invalid values)")
            except Exception as e:
                insights.append(f"⚠️ **Revenue Error**: {str(e)}")

        # Total Units Sold
        if detected["units_sold"]:
            try:
                units_data = pd.to_numeric(df[detected["units_sold"]], errors='coerce')
                total_units = units_data.sum()
                if pd.notna(total_units):
                    insights.append(f"📦 **Total Units Sold**: {int(total_units):,}")
                else:
                    insights.append("⚠️ **Units Data**: Could not calculate total units (invalid values)")
            except Exception as e:
                insights.append(f"⚠️ **Units Error**: {str(e)}")

        # Average Price
        if detected["price"]:
            try:
                price_data = pd.to_numeric(df[detected["price"]], errors='coerce')
                avg_price = price_data.mean()
                if pd.notna(avg_price):
                    insights.append(f"💲 **Average Price**: ${avg_price:.2f}")
            except Exception as e:
                insights.append(f"⚠️ **Price Error**: {str(e)}")

        # Top Product
        if detected["product"]:
            try:
                mode_result = df[detected["product"]].mode()
                if len(mode_result) > 0:
                    top_product = mode_result[0]
                    count = df[detected["product"]].value_counts().iloc[0]
                    insights.append(f"🏆 **Top Product**: *{top_product}* ({count} sales)")
                else:
                    insights.append("⚠️ **Product Data**: No clear top product found")
            except Exception as e:
                insights.append(f"⚠️ **Product Error**: {str(e)}")

        # Top Region
        if detected["region"]:
            try:
                mode_result = df[detected["region"]].mode()
                if len(mode_result) > 0:
                    top_region = mode_result[0]
                    count = df[detected["region"]].value_counts().iloc[0]
                    insights.append(f"🌍 **Top Region**: *{top_region}* ({count} sales)")
                else:
                    insights.append("⚠️ **Region Data**: No clear top region found")
            except Exception as e:
                insights.append(f"⚠️ **Region Error**: {str(e)}")

        # Top Platform
        if detected["platform"]:
            try:
                mode_result = df[detected["platform"]].mode()
                if len(mode_result) > 0:
                    top_platform = mode_result[0]
                    count = df[detected["platform"]].value_counts().iloc[0]
                    insights.append(f"🛒 **Top Platform**: *{top_platform}* ({count} sales)")
                else:
                    insights.append("⚠️ **Platform Data**: No clear top platform found")
            except Exception as e:
                insights.append(f"⚠️ **Platform Error**: {str(e)}")

        # Time Range & Seasonality
        if detected["date"]:
            try:
                df[detected["date"]] = pd.to_datetime(df[detected["date"]], errors='coerce')
                valid_dates = df[detected["date"]].dropna()
                if len(valid_dates) > 0:
                    earliest = valid_dates.min().strftime("%b %Y")
                    latest = valid_dates.max().strftime("%b %Y")
                    insights.append(f"📅 **Time Range**: {earliest} to {latest}")

                    # Peak sales month (if revenue available)
                    if detected["revenue"]:
                        df['Month'] = valid_dates.dt.to_period("M")
                        monthly_revenue = df.groupby("Month")[detected["revenue"]].sum()
                        if len(monthly_revenue) > 0:
                            peak_month = monthly_revenue.idxmax().strftime("%b %Y")
                            insights.append(f"📈 **Peak Sales Month**: {peak_month}")
                else:
                    insights.append("⚠️ **Date Data**: No valid dates found")
            except Exception as e:
                insights.append(f"⚠️ **Date Parsing Error**: {str(e)}")
        else:
            insights.append("📅 **Date Column**: Not detected — add a date column for time trends")

        # Data Quality
        try:
            missing = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (missing / total_cells) * 100
            if missing == 0:
                insights.append("✅ **Data Quality**: Clean — no missing values")
            else:
                insights.append(f"⚠️ **Data Quality**: {missing} missing values found ({missing_pct:.1f}%)")
        except Exception as e:
            insights.append(f"⚠️ **Data Quality Check Failed**: {str(e)}")

        # --- Discount Impact Analysis ---
        if "Discount" in df.columns:
            try:
                avg_discount = df["Discount"].mean()
                high_discount_sales = len(df[df["Discount"] > 0.2])
                revenue_with_discount = df[df["Discount"] > 0]["Revenue"].sum()
                revenue_without = df[df["Discount"] == 0]["Revenue"].sum()

                insights.append(f"🎁 **Average Discount**: {avg_discount:.1%}")
                insights.append(f"📈 **High Discount Sales**: {high_discount_sales} sales at >20% off")
                if revenue_with_discount > revenue_without:
                    insights.append("💡 **Discount Strategy Working**: Discounted items generate more revenue")
            except Exception as e:
                insights.append(f"⚠️ **Discount Analysis Error**: {str(e)}")

        # --- Business Recommendations ---
        try:
            recommendations = []

            # Promote top seller
            if detected["product"]:
                top_product = df[detected["product"]].mode()[0] if not df[detected["product"]].mode().empty else None
                if top_product:
                    recommendations.append(f"💡 Promote *{top_product}* — it's your top seller")

            # Use discounts strategically
            if "Discount" in df.columns and df["Discount"].mean() > 0.15:
                recommendations.append("💡 Use discounts strategically — they boost volume")

            # Prepare for peak season
            if detected["date"] and detected["revenue"]:
                df[detected["date"]] = pd.to_datetime(df[detected["date"]], errors='coerce')
                df['Month'] = df[detected["date"]].dt.month
                peak_month = df.groupby("Month")["Revenue"].sum().idxmax()
                if peak_month in [11, 12, 1]:
                    recommendations.append("📅 Plan inventory for holiday season (Nov–Jan)")

            # Add to insights
            if recommendations:
                insights.append("💼 **Business Recommendations**")
                insights.extend(recommendations)
        except Exception as e:
            insights.append(f"⚠️ **Recommendations Error**: {str(e)}")

        # --- Generate EDA Report ---
        try:
            # ✅ Use only valid parameters
            profile = ProfileReport(
                df,
                title=f"DataWise Report: {file.filename}",
                explorative=True,
                html={
                    "style": {
                        "primary_color": "#3498db",
                        "css": """
                            .intro { 
                                background: #f8f9fa; 
                                padding: 20px; 
                                border-radius: 8px; 
                                border-left: 4px solid #3498db; 
                                margin: 20px 0; 
                                font-family: 'Segoe UI', sans-serif;
                            }
                            .intro h4 { margin: 0 0 10px 0; }
                            .intro ul { text-align: left; margin: 10px 0; padding-left: 20px; }
                            .intro li { margin: 5px 0; }
                        """
                    }
                }
            )

            # Generate raw HTML
            html_content = profile.to_html()

            # ✅ Inject Top Insights at the top
            intro_html = f"""
            <div class="intro">
                <h4>📊 DataWise Top Insights</h4>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in insights)}
                </ul>
            </div>
            """

            # Insert after <body>
            html_content = html_content.replace('<body>', f'<body>{intro_html}')

            # Save to file
            report_path = os.path.join(UPLOAD_FOLDER, "report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

        # Return insights to frontend
        return {
            "message": "File uploaded and analyzed successfully!",
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "report_path": report_path,
            "report_url": "/report",
            "download_html_url": "/download-html",
            "insights": insights
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # Show full error
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ==================================
# 📊 Serve the Generated Report
# ==================================
@app.get("/report")
def get_report():
    """
    Serve the interactive HTML report.
    The report includes embedded Top Insights.
    """
    report_path = os.path.join(UPLOAD_FOLDER, "report.html")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found. Please upload a file first.")
    return FileResponse(report_path)


# ==================================
# 💾 Download HTML Report
# ==================================
@app.get("/download-html")
def download_html():
    """
    Allow user to download the HTML report.
    Includes all embedded insights and charts.
    """
    report_path = os.path.join(UPLOAD_FOLDER, "report.html")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        path=report_path,
        filename="DataWise_Report.html",
        media_type="text/html"
    )