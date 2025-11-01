from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# optional: IsolationForest for multivariate anomalies
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "data"
STATIC_FOLDER = "static"
PLOTS_FOLDER = os.path.join(STATIC_FOLDER, "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)


# ---------- Utility: save plot helper ----------
def save_plot(fig, name):
    """
    Save a matplotlib figure to static/plots and return the relative path used in templates.
    """
    path = os.path.join(PLOTS_FOLDER, name)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=150)
    plt.close(fig)
    # return the path relative to static/
    return f"plots/{name}"


def make_placeholder(name, text="Not enough data to generate this plot"):
    """
    Create a placeholder PNG with a centered message (used when data is insufficient).
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, wrap=True)
    ax.axis("off")
    return save_plot(fig, name)


# ---------- Hybrid Insight Engine ----------
def analyze_dataframe(df):
    results = {}
    n_rows, n_cols = df.shape

    # Basic structural info
    results["n_rows"] = int(n_rows)
    results["n_cols"] = int(n_cols)
    results["columns"] = list(df.columns)

    # Data types & counts
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    results["data_types"] = data_types

    # Missing & duplicates
    total_cells = n_rows * n_cols if n_rows * n_cols > 0 else 1
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = round(missing_cells / total_cells * 100, 2)
    duplicate_rows = int(df.duplicated().sum())

    results["missing_cells"] = missing_cells
    results["missing_pct"] = missing_pct
    results["duplicate_rows"] = duplicate_rows

    # Data quality score (0-100)
    completeness = df.notnull().mean().mean() * 100  # average column completeness
    uniqueness = df.drop_duplicates().shape[0] / n_rows * 100 if n_rows > 0 else 0
    relevance = (1 - sum(df.nunique() <= 1) / max(1, n_cols)) * 100  # % non-constant
    quality_score = round(0.4 * completeness + 0.3 * uniqueness + 0.3 * relevance, 2)
    results["quality_score"] = quality_score

    # Numeric and categorical columns
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    cat_df = df.select_dtypes(exclude=[np.number])
    categorical_cols = cat_df.columns.tolist()
    results["numeric_cols"] = numeric_cols
    results["categorical_cols"] = categorical_cols

    # ---------- Outliers & anomaly scoring ----------
    outlier_summary = {}
    if not numeric_df.empty:
        # IQR per column
        iqr_outliers = {}
        for col in numeric_cols:
            col_vals = numeric_df[col].dropna()
            if col_vals.empty:
                iqr_outliers[col] = 0.0
                continue
            q1 = col_vals.quantile(0.25)
            q3 = col_vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            pct_out = ((col_vals < lower) | (col_vals > upper)).mean() * 100
            iqr_outliers[col] = round(pct_out, 2)
        outlier_summary["iqr_outliers_pct"] = iqr_outliers

        # Z-score method (extreme)
        z_outliers_pct = {}
        zs = np.abs(stats.zscore(numeric_df, nan_policy="omit"))
        if zs.size != 0:
            if zs.ndim == 1:
                z_outliers_pct[numeric_cols[0]] = round((zs > 3).mean() * 100, 2)
            else:
                for i, col in enumerate(numeric_cols):
                    col_z = zs[:, i]
                    # ignore nan
                    col_mask = ~np.isnan(col_z)
                    if col_mask.sum() == 0:
                        z_outliers_pct[col] = 0.0
                    else:
                        z_outliers_pct[col] = round((col_z[col_mask] > 3).mean() * 100, 2)
        outlier_summary["z_outliers_pct"] = z_outliers_pct

        # Multivariate anomaly via IsolationForest if available
        if SKLEARN_AVAILABLE and len(numeric_cols) >= 2:
            try:
                iso = IsolationForest(contamination="auto", random_state=42)
                mask = iso.fit_predict(numeric_df.fillna(numeric_df.mean()))
                iso_out_pct = round((mask == -1).mean() * 100, 2)
            except Exception:
                iso_out_pct = None
        else:
            iso_out_pct = None
        outlier_summary["isolation_forest_outliers_pct"] = iso_out_pct
    results["outlier_summary"] = outlier_summary

    # ---------- Distribution (skewness) & transform recommendations ----------
    skew_recs = {}
    cv_scores = {}
    if not numeric_df.empty:
        skewness = numeric_df.skew().fillna(0)
        for col in numeric_cols:
            s = float(skewness.get(col, 0.0))
            if abs(s) > 2:
                rec = "Box-Cox or log transform"
            elif abs(s) > 1:
                rec = "Log transform (log1p if zeros)"
            elif abs(s) > 0.5:
                rec = "Square root transform"
            else:
                rec = "No transform needed"
            skew_recs[col] = {"skewness": round(s, 3), "recommendation": rec}
            # CV
            col_mean = numeric_df[col].mean()
            col_std = numeric_df[col].std()
            cv = (col_std / col_mean * 100) if col_mean != 0 else np.nan
            cv_scores[col] = round(float(cv) if not np.isnan(cv) else 0.0, 2)
    results["skew_recs"] = skew_recs
    results["cv_scores"] = cv_scores

    # ---------- Class imbalance detection (categorical) ----------
    imbalance = {}
    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True, dropna=False)
        if len(vc) == 0:
            continue
        top_ratio = float(vc.iloc[0])
        imbalance[col] = round(top_ratio * 100, 2)
    results["imbalance_pct_top"] = imbalance

    # ---------- Feature importance composite score ----------
    importance_scores = {}
    for col in df.columns:
        if col in cv_scores:
            cv = cv_scores[col] / 100.0
        else:
            cv = 0.0
        uniqueness = df[col].nunique() / max(1, n_rows)
        completeness_col = df[col].notnull().mean()
        has_outliers = 0.0
        if col in outlier_summary.get("iqr_outliers_pct", {}):
            has_outliers = min(1.0, outlier_summary["iqr_outliers_pct"][col] / 100.0)
        score = 0.3 * cv + 0.3 * uniqueness + 0.2 * completeness_col + 0.2 * has_outliers
        importance_scores[col] = round(score * 100, 2)
    results["importance_scores"] = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

    # ---------- Correlation & redundancy analysis ----------
    corr_insights = {}
    if len(numeric_cols) >= 2:
        corr = numeric_df.corr().abs()
        pairs = corr.unstack().sort_values(ascending=False)
        pairs = pairs[pairs < 1.0]
        if not pairs.empty:
            top_pair = pairs.index[0]
            corr_insights["top_correlation"] = {"pair": list(top_pair), "value": round(float(pairs.iloc[0]), 3)}
        redundant = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                a = numeric_cols[i]
                b = numeric_cols[j]
                if corr.loc[a, b] > 0.9:
                    redundant.append((a, b, round(float(corr.loc[a, b]), 3)))
        corr_insights["redundant_pairs"] = redundant
    results["corr_insights"] = corr_insights

    # ---------- Temporal detection ----------
    temporal_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "dt", "timestamp"))]
    results["temporal_cols"] = temporal_cols

    # ---------- Cardinality recommendations ----------
    cardinality = {}
    for col in df.columns:
        uniq = df[col].nunique(dropna=False)
        if uniq < 10:
            rec = "Low cardinality: use one-hot or binary encoding"
        elif uniq < 50:
            rec = "Medium cardinality: consider frequency or target encoding"
        elif uniq < 0.95 * n_rows:
            rec = "High cardinality: embedding or hash features"
        else:
            rec = "Very high unique: likely identifier — exclude from modeling"
        cardinality[col] = {"unique": int(uniq), "recommendation": rec}
    results["cardinality"] = cardinality

    # ---------- Missing pattern hint (basic) ----------
    missing_pattern = {}
    miss_perc = (df.isnull().mean() * 100).round(2).to_dict()
    missing_pattern["flags"] = {col: pct for col, pct in miss_perc.items() if pct > 20}
    results["missing_pattern"] = missing_pattern

    # ---------- Rare event detection ----------
    rare_events = {}
    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True, dropna=False)
        if not vc.empty and vc.iloc[-1] < 0.10:
            rare_events[col] = vc.tail().to_dict()
    rare_numeric = {}
    for col in numeric_cols:
        vals = numeric_df[col].dropna()
        if vals.empty:
            continue
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        rare_pct = ((vals < lower) | (vals > upper)).mean() * 100
        if rare_pct > 0:
            rare_numeric[col] = round(rare_pct, 2)
    rare_events["numeric_extremes_pct"] = rare_numeric
    results["rare_events"] = rare_events

    return results


# ---------- Visualization generator ----------
def generate_plots(df, results):
    plot_refs = {}

    numeric_cols = results.get("numeric_cols", [])
    # Correlation heatmap
    try:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            plot_refs["heatmap"] = save_plot(fig, "correlation_heatmap.png")
        else:
            plot_refs["heatmap"] = make_placeholder("correlation_heatmap.png", "Not enough numeric columns for a correlation heatmap.")
    except Exception:
        plot_refs["heatmap"] = make_placeholder("correlation_heatmap.png", "Unable to generate heatmap.")

    # Trends (if temporal detected)
    temporal_cols = results.get("temporal_cols", [])
    try:
        if temporal_cols and len(numeric_cols) >= 1:
            tcol = temporal_cols[0]
            tmp = df.copy()
            tmp[tcol] = pd.to_datetime(tmp[tcol], errors="coerce")
            tmp = tmp.sort_values(tcol).dropna(subset=[tcol])
            if tmp.shape[0] >= 2:
                cols = numeric_cols[:3]
                fig, ax = plt.subplots(figsize=(8, 4))
                for c in cols:
                    ax.plot(tmp[tcol], tmp[c], label=c)
                ax.set_title("Trends (first temporal column)")
                ax.legend()
                plot_refs["trends"] = save_plot(fig, "trends.png")
            else:
                plot_refs["trends"] = make_placeholder("trends.png", "Not enough temporal data to plot trends.")
        else:
            plot_refs["trends"] = make_placeholder("trends.png", "No temporal column detected for trend plot.")
    except Exception:
        plot_refs["trends"] = make_placeholder("trends.png", "Unable to generate trend plot.")

    # Outlier summary plot (boxplot for top 3 numeric columns by CV)
    try:
        cv_scores = results.get("cv_scores", {})
        if cv_scores:
            top_by_cv = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            cols = [c for c, _ in top_by_cv if c in df.columns]
            if cols:
                fig, ax = plt.subplots(figsize=(7, 4))
                df[cols].boxplot(ax=ax)
                ax.set_title("Boxplots (top CV columns)")
                plot_refs["outliers_box"] = save_plot(fig, "outliers_box.png")
            else:
                plot_refs["outliers_box"] = make_placeholder("outliers_box.png", "Not enough numeric data for outlier boxplots.")
        else:
            plot_refs["outliers_box"] = make_placeholder("outliers_box.png", "Not enough numeric data for outlier boxplots.")
    except Exception:
        plot_refs["outliers_box"] = make_placeholder("outliers_box.png", "Unable to generate outlier plots.")

    # Feature importance bar
    try:
        importance = results.get("importance_scores", {})
        if importance:
            names = list(importance.keys())[:10]
            values = [importance[n] for n in names]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(list(reversed(names)), list(reversed(values)))
            ax.set_xlabel("Importance score")
            ax.set_title("Top Features by Composite Importance")
            plot_refs["feature_importance"] = save_plot(fig, "feature_importance.png")
        else:
            plot_refs["feature_importance"] = make_placeholder("feature_importance.png", "No feature importance information available.")
    except Exception:
        plot_refs["feature_importance"] = make_placeholder("feature_importance.png", "Unable to generate feature importance chart.")

    # Data quality bar
    try:
        fig, ax = plt.subplots(figsize=(6, 1.8))
        score = results.get("quality_score", 0)
        ax.barh([0], [score], color="#4c72b0")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Data Quality Score (0-100)")
        ax.text(min(score + 1, 95), 0, f"{score}", va="center")
        fig.suptitle("Data Quality Score")
        plot_refs["quality_score"] = save_plot(fig, "quality_score.png")
    except Exception:
        plot_refs["quality_score"] = make_placeholder("quality_score.png", "Unable to generate data quality score chart.")

    # Skewness bar
    try:
        skew_recs = results.get("skew_recs", {})
        if skew_recs:
            cols = list(skew_recs.keys())[:10]
            vals = [skew_recs[c]["skewness"] for c in cols]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(cols, vals)
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_ylabel("Skewness")
            ax.set_title("Skewness (top columns)")
            plot_refs["skewness"] = save_plot(fig, "skewness.png")
        else:
            plot_refs["skewness"] = make_placeholder("skewness.png", "No skewness information available.")
    except Exception:
        plot_refs["skewness"] = make_placeholder("skewness.png", "Unable to generate skewness chart.")

    return plot_refs


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return "Invalid file. Please upload a CSV.", 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    session["filepath"] = filepath
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for("index"))

    df = pd.read_csv(filepath, encoding="latin1")
    preview = df.head(10).to_html(classes="table table-striped table-bordered", index=False)

    # compute structural stats
    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(exclude=[np.number]).columns)
    missing_cells = int(df.isnull().sum().sum())
    missing_percent = round((missing_cells / (df.shape[0] * df.shape[1])) * 100, 2)

    return render_template(
        "dashboard.html",
        preview=preview,
        total_rows=total_rows,
        total_cols=total_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_percent=missing_percent,
        filename=os.path.basename(filepath),
    )


@app.route("/analysis")
def analysis():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for("index"))

    df = pd.read_csv(filepath, encoding="latin1")
    results = analyze_dataframe(df)
    plot_refs = generate_plots(df, results)

    # Prepare human-readable insights (short list prioritized by business value)
    insights = []

    # Executive summary lines
    insights.append(f"Data quality score: {results.get('quality_score')} (higher is better).")
    insights.append(f"Total rows: {results.get('n_rows')}, Total columns: {results.get('n_cols')}.")
    if results.get("missing_pct", 0) > 0:
        insights.append(f"Missing data: {results.get('missing_pct')}% of all cells are missing.")
    if results.get("duplicate_rows", 0) > 0:
        insights.append(f"Duplicate rows detected: {results.get('duplicate_rows')}.")

    # Outliers & anomalies
    out_summary = results.get("outlier_summary", {})
    if out_summary:
        iqr_map = out_summary.get("iqr_outliers_pct", {})
        top_outliers = sorted(iqr_map.items(), key=lambda x: x[1], reverse=True)[:3]
        for col, pct in top_outliers:
            if pct > 0:
                insights.append(f"Outlier note: {col} has {pct}% values outside IQR bounds (possible anomalies).")
        iso_pct = out_summary.get("isolation_forest_outliers_pct")
        if iso_pct:
            insights.append(f"IsolationForest flagged {iso_pct}% of records as multivariate anomalies (if available).")

    # Skewness & transform recs
    skew_recs = results.get("skew_recs", {})
    skew_summaries = []
    for col, info in list(skew_recs.items())[:5]:
        skew_summaries.append(f"{col} skew={info['skewness']}: {info['recommendation']}")
    if skew_summaries:
        insights.append("Transformation suggestions: " + "; ".join(skew_summaries))

    # Imbalance detection (categorical)
    imbal = results.get("imbalance_pct_top", {})
    for col, top_pct in imbal.items():
        if top_pct > 90:
            insights.append(f"Class imbalance: {col} majority class comprises {top_pct}% — consider rebalancing for ML tasks.")

    # Feature importance top
    importance = results.get("importance_scores", {})
    if importance:
        top_features = list(importance.items())[:5]
        insights.append("Top features by composite importance: " + ", ".join([f"{k} ({v})" for k, v in top_features]))

    # Correlation redundancy
    redund = results.get("corr_insights", {}).get("redundant_pairs", [])
    if redund:
        insights.append(f"Redundant feature pairs detected (r>0.9): " + ", ".join([f"{a}/{b} ({r})" for a, b, r in redund[:5]]))

    # Rare events
    rare = results.get("rare_events", {})
    if rare.get("numeric_extremes_pct"):
        for c, p in rare["numeric_extremes_pct"].items():
            insights.append(f"Rare extreme values in {c}: {p}% of observations are extreme.")

    # Final actionable recommendations (short)
    recommendations = []
    if results.get("quality_score", 0) < 80:
        recommendations.append("Data quality below 80: consider cleaning missing values, removing duplicates, or collecting more data.")
    if redund:
        recommendations.append("Remove or merge redundant features to avoid multicollinearity and simplify models.")
    if any(v > 10 for v in results.get("cv_scores", {}).values()):
        recommendations.append("High CV detected for some features — they are strong candidates for predictive models.")

    # generate stats and missing counts for template
    stats_html = df.describe(include="all").round(4).to_html(classes="table table-bordered table-hover")
    null_counts_html = df.isnull().sum().to_frame("Missing Values").reset_index().to_html(classes="table table-bordered", index=False)
    unique_counts_html = df.nunique().to_frame("Unique Values").reset_index().to_html(classes="table table-bordered", index=False)

    return render_template(
        "analysis.html",
        insights=insights,
        recommendations=recommendations,
        stats=stats_html,
        null_counts=null_counts_html,
        unique_counts=unique_counts_html,
        plots=plot_refs,
        filename=os.path.basename(filepath),
    )


if __name__ == "__main__":
    app.run(debug=True)
