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

        # --- Detect Data Type ---
        def detect_data_type(df):
            """Guess the data type based on column names"""
            text = " ".join(df.columns).lower()
            if any(kw in text for kw in ["score", "grade", "exam", "test", "student", "study", "absence"]):
                return "education"
            elif any(kw in text for kw in ["revenue", "sales", "price", "product", "units", "order", "date"]):
                return "sales"
            else:
                return "generic"

        data_type = detect_data_type(df)

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

        if data_type == "education":
            # --- Define subject columns ---
            subject_cols = ["math_score", "history_score", "physics_score", "chemistry_score", 
                          "biology_score", "english_score", "geography_score"]
            valid_subjects = [col for col in subject_cols if col in df.columns]
            
            if len(valid_subjects) > 0:
                try:
                    # Calculate total and average scores
                    df["total_score"] = df[valid_subjects].sum(axis=1)
                    df["avg_score"] = df[valid_subjects].mean(axis=1)

                    # 🏆 Top Academic Performer (highest average)
                    top_avg_idx = df["avg_score"].idxmax()
                    top_avg_name = f"{df.loc[top_avg_idx, 'first_name']} {df.loc[top_avg_idx, 'last_name']}"
                    top_avg = df.loc[top_avg_idx, "avg_score"]
                    insights.append(f"🏆 **Top Academic Performer**: *{top_avg_name}* (Avg: {top_avg:.1f})")

                    # 📊 Top Subject Scorer (e.g., Math)
                    if "math_score" in df.columns:
                        top_math_idx = df["math_score"].idxmax()
                        top_math_name = f"{df.loc[top_math_idx, 'first_name']} {df.loc[top_math_idx, 'last_name']}"
                        top_math = df.loc[top_math_idx, "math_score"]
                        insights.append(f"📊 **Top Math Scorer**: *{top_math_name}* (Math: {top_math})")

                    # ⚖️ Most Balanced Performer (lowest score std)
                    df["score_std"] = df[valid_subjects].std(axis=1)
                    most_balanced_idx = df["score_std"].idxmin()
                    most_balanced_name = f"{df.loc[most_balanced_idx, 'first_name']} {df.loc[most_balanced_idx, 'last_name']}"
                    avg_balanced = df.loc[most_balanced_idx, "avg_score"]
                    insights.append(f"⚖️ **Most Balanced Performer**: *{most_balanced_name}* (Avg: {avg_balanced:.1f})")

                    # 📈 Subject Averages
                    subject_means = df[valid_subjects].mean().sort_values(ascending=False)
                    highest = subject_means.index[0]
                    lowest = subject_means.index[-1]
                    insights.append(f"📈 **Strongest Subject**: *{highest.replace('_score', '').title()}* (Avg: {subject_means.iloc[0]:.1f})")
                    insights.append(f"📉 **Weakest Subject**: *{lowest.replace('_score', '').title()}* (Avg: {subject_means.iloc[-1]:.1f})")

                except Exception as e:
                    insights.append(f"⚠️ **Performance Analysis Error**: {str(e)}")

            # --- Study Habits vs Performance ---
            if "weekly_self_study_hours" in df.columns and "math_score" in df.columns:
                try:
                    high_study = df[df["weekly_self_study_hours"] > 20]
                    low_study = df[df["weekly_self_study_hours"] <= 10]
                    if len(high_study) > 0 and len(low_study) > 0:
                        high_avg = high_study["math_score"].mean()
                        low_avg = low_study["math_score"].mean()
                        improvement = ((high_avg - low_avg) / low_avg * 100) if low_avg > 0 else 0
                        if improvement > 10:
                            insights.append(f"💡 Students who study >20 hrs/week score **{improvement:.0f}% higher** in math")
                except Exception as e:
                    pass

            # Top performers' study hours
            if "weekly_self_study_hours" in df.columns and "avg_score" in df.columns:
                try:
                    top_10_percent = df.nlargest(int(len(df) * 0.1), "avg_score")
                    avg_study_top = top_10_percent["weekly_self_study_hours"].mean()
                    if avg_study_top > 30:
                        insights.append(f"💡 Top 10% students study **{avg_study_top:.0f} hrs/week** on average")
                except Exception as e:
                    pass

            # Part-time job impact
            if "part_time_job" in df.columns and "math_score" in df.columns:
                try:
                    with_job = df[df["part_time_job"] == True]["math_score"].mean()
                    without_job = df[df["part_time_job"] == False]["math_score"].mean()
                    if with_job < without_job:
                        diff = without_job - with_job
                        insights.append(f"📉 Students with part-time jobs score **{diff:.1f} points lower** in math")
                except Exception as e:
                    pass

            # Gender & Performance Trends
            if "gender" in df.columns:
                try:
                    # STEM: Math, Physics, Chemistry
                    stem_cols = [c for c in ["math_score", "physics_score", "chemistry_score"] if c in df.columns]
                    if len(stem_cols) > 0:
                        df["stem_avg"] = df[stem_cols].mean(axis=1)
                        male_stem = df[df["gender"] == "male"]["stem_avg"].mean()
                        female_stem = df[df["gender"] == "female"]["stem_avg"].mean()
                        if male_stem > female_stem + 3:
                            insights.append("📊 Boys outperform girls in STEM subjects")
                        elif female_stem > male_stem + 3:
                            insights.append("📊 Girls outperform boys in STEM subjects")

                    # Humanities: History, English, Geography
                    hum_cols = [c for c in ["history_score", "english_score", "geography_score"] if c in df.columns]
                    if len(hum_cols) > 0:
                        df["hum_avg"] = df[hum_cols].mean(axis=1)
                        male_hum = df[df["gender"] == "male"]["hum_avg"].mean()
                        female_hum = df[df["gender"] == "female"]["hum_avg"].mean()
                        if female_hum > male_hum + 3:
                            insights.append("📘 Girls outperform boys in humanities")
                        elif male_hum > female_hum + 3:
                            insights.append("📘 Boys outperform boys in humanities")

                    # Study hours by gender
                    if "weekly_self_study_hours" in df.columns:
                        male_study = df[df["gender"] == "male"]["weekly_self_study_hours"].mean()
                        female_study = df[df["gender"] == "female"]["weekly_self_study_hours"].mean()
                        if abs(male_study - female_study) > 5:
                            higher = "boys" if male_study > female_study else "girls"
                            insights.append(f"📘 {higher.title()} study more per week")
                except Exception as e:
                    pass

        else:
            # 🛒 Sales-Specific Insights
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

                        if detected["revenue"]:
                            df['Month'] = valid_dates.dt.to_period("M")
                            monthly_revenue = df.groupby("Month")[detected["revenue"]].sum()
                            if len(monthly_revenue) > 0:
                                peak_month = monthly_revenue.idxmax().strftime("%b %Y")
                                insights.append(f"📈 **Peak Sales Month**: {peak_month}")
                except Exception as e:
                    insights.append(f"⚠️ **Date Parsing Error**: {str(e)}")

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

            # --- Smart Sales Intelligence ---
            try:
                # Holiday season peak
                if detected["date"] and detected["revenue"]:
                    df[detected["date"]] = pd.to_datetime(df[detected["date"]], errors='coerce')
                    df['Month'] = df[detected["date"]].dt.month
                    monthly_revenue = df.groupby("Month")[detected["revenue"]].sum()
                    peak_month = monthly_revenue.idxmax()
                    if peak_month in [11, 12, 1]:
                        insights.append("🎄 **Holiday Season Peak**: Sales spike in Q4 — prepare inventory")
                    elif peak_month in [6, 7, 8]:
                        insights.append("☀️ **Summer Peak**: Strong sales in summer — consider seasonal promotions")

                # Price vs. Demand
                if detected["price"] and detected["units_sold"]:
                    try:
                        corr = df[detected["price"]].corr(pd.to_numeric(df[detected["units_sold"]], errors='coerce'))
                        if corr < -0.3:
                            insights.append("📉 **Price Sensitivity**: Higher prices → lower sales (elastic demand)")
                        elif corr > 0.3:
                            insights.append("📈 **Premium Effect**: Higher prices → higher sales (luxury product?)")
                    except Exception as e:
                        pass

                # Platform Performance
                if detected["platform"] and detected["revenue"]:
                    try:
                        platform_perf = df.groupby(detected["platform"])[detected["revenue"]].sum().sort_values(ascending=False)
                        top_platform = platform_perf.index[0]
                        dominance = (platform_perf.iloc[0] / platform_perf.sum()) * 100
                        if dominance > 50:
                            insights.append(f"🚀 **{top_platform} Dominates**: {dominance:.0f}% of revenue — focus ad spend here")
                    except Exception as e:
                        pass

            except Exception as e:
                insights.append(f"⚠️ **Smart Insights Error**: {str(e)}")

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
                if detected["product"]:
                    top_product = df[detected["product"]].mode()[0] if not df[detected["product"]].mode().empty else None
                    if top_product:
                        recommendations.append(f"💡 Promote *{top_product}* — it's your top seller")
                if "Discount" in df.columns and df["Discount"].mean() > 0.15:
                    recommendations.append("💡 Use discounts strategically — they boost volume")
                if detected["date"] and detected["revenue"]:
                    df[detected["date"]] = pd.to_datetime(df[detected["date"]], errors='coerce')
                    df['Month'] = df[detected["date"]].dt.month
                    peak_month = df.groupby("Month")["Revenue"].sum().idxmax()
                    if peak_month in [11, 12, 1]:
                        recommendations.append("📅 Plan inventory for holiday season (Nov–Jan)")
                if recommendations:
                    insights.append("💼 **Business Recommendations**")
                    insights.extend(recommendations)
            except Exception as e:
                insights.append(f"⚠️ **Recommendations Error**: {str(e)}")

        # --- Generate EDA Report ---
        try:
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

            html_content = profile.to_html()

            intro_html = f"""
            <div class="intro">
                <h4>📊 DataWise Top Insights</h4>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in insights)}
                </ul>
            </div>
            """

            html_content = html_content.replace('<body>', f'<body>{intro_html}')

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
        traceback.print_exc()
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