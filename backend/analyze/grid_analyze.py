import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings('ignore')

INPUT_CSV = "grid_search_results.csv"
OUTPUT_DIR = "wyniki_magisterka"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "wykresy")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tabele")

VALID_CATEGORIES = [
    "rec.autos",
    "sci.space",
    "comp.graphics",
    "misc.forsale",
    "sci.med",
]

ALL_LABELS_WITH_ERROR = VALID_CATEGORIES + ["inne_blad"]
COT_PROMPTS = {"P4_ChainOfThought", "P6_FewShot_CoT"}

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
})


def clean_predictions(df):
    df = df.copy()
    df["predicted_label"] = df["predicted_label"].astype(str)

    def extract_label(row):
        text = str(row["predicted_label"]).lower()
        prompt_type = row["prompt_type"]

        if prompt_type in COT_PROMPTS:
            for line in reversed(text.splitlines()):
                line = line.strip()
                if "wynik:" in line:
                    candidate = line.split("wynik:")[-1].strip().lower()
                    for label in VALID_CATEGORIES:
                        if label in candidate:
                            return label

        for label in VALID_CATEGORIES:
            if label in text:
                return label

        return "inne_blad"

    df["predicted_clean"] = df.apply(extract_label, axis=1)
    return df


def classify_family(model_name):
    m = str(model_name).lower()

    if "phi" in m:
        return "Phi"
    if "llama" in m:
        return "Llama"
    if "gemma" in m:
        return "Gemma"

    return "Other"


def classify_type(model_name):
    return (
        "Fine-Tuned"
        if "tuned" in str(model_name).lower()
        else "Base"
    )


def safe_filename(text):
    return (
        str(text)
        .replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def generate_master_thesis_analysis():
    print(f"Loading data from file: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, sep=",", engine="python")
        df.columns = [c.strip().replace(";", "") for c in df.columns]
        df = df.apply(
            lambda col: col.map(
                lambda x: str(x).rstrip(";").strip()
            )
            if col.dtype == object
            else col
        )
    except FileNotFoundError:
        print(f"Error: file not found: {INPUT_CSV}")
        return

    df = clean_predictions(df)

    # ---------------------------------------------------------
    # Table 1: Class distribution
    # ---------------------------------------------------------

    class_dist = (
        df[["email_id", "true_label"]]
        .drop_duplicates()["true_label"]
        .value_counts()
        .reset_index()
    )

    class_dist.columns = ["Class", "Sample_Count"]

    class_dist["Share_Percent"] = (
        class_dist["Sample_Count"]
        / class_dist["Sample_Count"].sum()
        * 100
    ).round(2)

    class_dist.to_csv(
        os.path.join(TABLES_DIR, "tabela_1_rozklad_klas.csv"),
        index=False,
    )

    # ---------------------------------------------------------
    # Main evaluation metrics
    # ---------------------------------------------------------

    metrics = []
    reports_all = []

    for (model, prompt), group in df.groupby(
        ["model", "prompt_type"]
    ):
        y_true = group["true_label"]
        y_pred = group["predicted_clean"]

        metrics.append({
            "Model": model,
            "Prompt": prompt,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro_Precision": precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            ),
            "Macro_Recall": recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            ),
            "Macro_F1": f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            ),
            "Weighted_F1": f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            ),
            "Hallucination_Rate": (y_pred == "inne_blad").mean(),
            "Model_Family": classify_family(model),
            "Model_Type": classify_type(model),
        })

        report = classification_report(
            y_true,
            y_pred,
            labels=VALID_CATEGORIES + ["inne_blad"],
            output_dict=True,
            zero_division=0,
        )

        for label, values in report.items():
            if isinstance(values, dict):
                reports_all.append({
                    "Model": model,
                    "Prompt": prompt,
                    "Class": (
                        "other_error"
                        if label == "inne_blad"
                        else label
                    ),
                    "Precision": values.get("precision", 0),
                    "Recall": values.get("recall", 0),
                    "F1_Score": values.get("f1-score", 0),
                    "Support": values.get("support", 0),
                })

    metrics_df = (
        pd.DataFrame(metrics)
        .sort_values(by="Macro_F1", ascending=False)
        .reset_index(drop=True)
    )

    reports_df = pd.DataFrame(reports_all)

    metrics_df.to_csv(
        os.path.join(TABLES_DIR, "tabela_2_metryki_glowne.csv"),
        index=False,
    )

    reports_df.to_csv(
        os.path.join(TABLES_DIR, "tabela_3_metryki_per_klasa.csv"),
        index=False,
    )

    # Copy with English legend titles, used only for charts
    # (seaborn takes legend headings from column names).
    plot_df = metrics_df.rename(columns={
        "Model_Family": "Model family",
        "Model_Type": "Model type",
    })

    # ---------------------------------------------------------
    # Table 4: Base vs. fine-tuned model comparison
    # ---------------------------------------------------------

    avg_by_family_type = (
        metrics_df.groupby(["Model_Family", "Model_Type"])[
            [
                "Accuracy",
                "Macro_F1",
                "Weighted_F1",
                "Hallucination_Rate",
            ]
        ]
        .mean()
        .reset_index()
    )

    avg_by_family_type.to_csv(
        os.path.join(
            TABLES_DIR,
            "tabela_4_srednie_bazowy_vs_tuned.csv",
        ),
        index=False,
    )

    # ---------------------------------------------------------
    # Table 5: Fine-tuning deltas
    # ---------------------------------------------------------

    delta_rows = []

    for family in avg_by_family_type["Model_Family"].unique():
        fam = avg_by_family_type[
            avg_by_family_type["Model_Family"] == family
        ]

        base = fam[fam["Model_Type"] == "Base"]
        tuned = fam[fam["Model_Type"] == "Fine-Tuned"]

        if not base.empty and not tuned.empty:
            delta_rows.append({
                "Model_Family": family,
                "Delta_Accuracy": round(
                    float(tuned["Accuracy"].iloc[0])
                    - float(base["Accuracy"].iloc[0]),
                    4,
                ),
                "Delta_Macro_F1": round(
                    float(tuned["Macro_F1"].iloc[0])
                    - float(base["Macro_F1"].iloc[0]),
                    4,
                ),
                "Delta_Weighted_F1": round(
                    float(tuned["Weighted_F1"].iloc[0])
                    - float(base["Weighted_F1"].iloc[0]),
                    4,
                ),
                "Delta_Hallucination_Rate": round(
                    float(tuned["Hallucination_Rate"].iloc[0])
                    - float(base["Hallucination_Rate"].iloc[0]),
                    4,
                ),
            })

    pd.DataFrame(delta_rows).to_csv(
        os.path.join(
            TABLES_DIR,
            "tabela_5_delta_tuned_vs_base.csv",
        ),
        index=False,
    )

    # ---------------------------------------------------------
    # Table 6: Final ranking
    # ---------------------------------------------------------

    ranking_df = metrics_df.copy()
    ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

    ranking_df.to_csv(
        os.path.join(TABLES_DIR, "tabela_6_ranking_koncowy.csv"),
        index=False,
    )

    # ---------------------------------------------------------
    # Chart 1: F1-Macro heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 8))

    pivot_f1 = metrics_df.pivot(
        index="Model",
        columns="Prompt",
        values="Macro_F1",
    )

    sns.heatmap(
        pivot_f1,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
    )

    plt.title("F1-Macro Heatmap: Models vs. Prompts")
    plt.xlabel("Prompt Type")
    plt.ylabel("Model")

    plt.savefig(
        os.path.join(PLOTS_DIR, "wykres_1_heatmapa_f1_macro.png")
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 2: Accuracy heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 8))

    pivot_acc = metrics_df.pivot(
        index="Model",
        columns="Prompt",
        values="Accuracy",
    )

    sns.heatmap(
        pivot_acc,
        annot=True,
        fmt=".3f",
        cmap="magma",
        linewidths=0.5,
    )

    plt.title("Accuracy Heatmap: Models vs. Prompts")
    plt.xlabel("Prompt Type")
    plt.ylabel("Model")

    plt.savefig(
        os.path.join(PLOTS_DIR, "wykres_2_heatmapa_accuracy.png")
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 3: F1-Macro across configurations
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 7))

    sns.barplot(
        data=metrics_df,
        x="Model",
        y="Macro_F1",
        hue="Prompt",
    )

    plt.title("F1-Macro Across All Models and Prompt Variants")
    plt.xlabel("Model")
    plt.ylabel("Macro F1-Score")
    plt.xticks(rotation=45)

    plt.ylim(0, 1.05)

    plt.legend(
        title="Prompt Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_3_barplot_f1_konfiguracje.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 4: Impact of fine-tuning
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=avg_by_family_type,
        x="Model_Family",
        y="Macro_F1",
        hue="Model_Type",
        palette=["#d95f02", "#1b9e77"],
    )

    plt.title(
        "Impact of Fine-Tuning on Macro F1-Score\n"
        "(Average Across All Prompt Variants)"
    )
    plt.xlabel("Model Family")
    plt.ylabel("Mean Macro F1-Score")
    plt.ylim(0, 1.05)

    plt.legend(title="Model Type")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_4_wplyw_finetuningu.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 5: Prompt sensitivity
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    sns.pointplot(
        data=metrics_df,
        x="Prompt",
        y="Macro_F1",
        hue="Model_Family",
    )

    plt.title("Model Architecture Sensitivity to Prompt Variations")
    plt.xlabel("Prompt Type")
    plt.ylabel("Macro F1-Score")
    plt.xticks(rotation=20)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Model Family")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_5_wplyw_promptow_linie.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 6: Hallucination rate
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 6))

    sns.barplot(
        data=metrics_df,
        x="Model",
        y="Hallucination_Rate",
        hue="Prompt",
        palette="rocket",
    )

    plt.title(
        "Hallucination Rate by Model and Prompt Configuration"
    )
    plt.xlabel("Model")
    plt.ylabel("Hallucination Rate")
    plt.xticks(rotation=45)

    plt.legend(
        title="Prompt Type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.savefig(
        os.path.join(PLOTS_DIR, "wykres_6_halucynacje.png")
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 7: F1-Macro vs. hallucination rate
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=plot_df,
        x="Hallucination_Rate",
        y="Macro_F1",
        hue="Model family",
        style="Model type",
        s=150,
    )

    plt.title(
        "Relationship Between Hallucination Rate and Macro F1-Score"
    )
    plt.xlabel("Hallucination Rate")
    plt.ylabel("Macro F1-Score")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_7_scatter_f1_vs_halucynacje.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 8: F1-Macro distribution
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=metrics_df,
        x="Model_Family",
        y="Macro_F1",
        hue="Model_Type",
        palette="pastel",
    )

    plt.title(
        "Macro F1-Score Distribution by Model Family\n"
        "(Across Prompt Variants)"
    )
    plt.xlabel("Model Family")
    plt.ylabel("Macro F1-Score")
    plt.legend(title="Model Type")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_8_boxplot_stabilnosc.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Best-performing configuration
    # ---------------------------------------------------------

    best_row = metrics_df.iloc[0]
    best_model = best_row["Model"]
    best_prompt = best_row["Prompt"]

    best_subset = df[
        (df["model"] == best_model)
        & (df["prompt_type"] == best_prompt)
    ]

    best_report = classification_report(
        best_subset["true_label"],
        best_subset["predicted_clean"],
        labels=VALID_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    best_class_rows = [
        {
            "Class": label,
            "Precision": best_report[label]["precision"],
            "Recall": best_report[label]["recall"],
            "F1_Score": best_report[label]["f1-score"],
            "Support": best_report[label]["support"],
        }
        for label in VALID_CATEGORIES
    ]

    best_class_df = pd.DataFrame(best_class_rows)

    best_class_df.to_csv(
        os.path.join(
            TABLES_DIR,
            "tabela_7_najlepszy_model_per_klasa.csv",
        ),
        index=False,
    )

    print("Table 7: Per-class metrics for the best model")

    # ---------------------------------------------------------
    # Chart 9: Best configuration metrics by class
    # ---------------------------------------------------------

    melted_best = best_class_df.melt(
        id_vars="Class",
        value_vars=["Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=melted_best,
        x="Class",
        y="Value",
        hue="Metric",
    )

    plt.title(
        "Precision, Recall and F1-Score by Class — Best Configuration\n"
        f"{best_model} | {best_prompt}"
    )
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_9_metryki_per_klasa_best.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Best and worst confusion matrices
    # ---------------------------------------------------------

    worst_row = metrics_df.iloc[-1]
    worst_model = worst_row["Model"]
    worst_prompt = worst_row["Prompt"]

    for model_name, prompt_name, desc, cmap_color in [
        (best_model, best_prompt, "best", "Blues"),
        (worst_model, worst_prompt, "worst", "Reds"),
    ]:
        subset = df[
            (df["model"] == model_name)
            & (df["prompt_type"] == prompt_name)
        ]

        cm = confusion_matrix(
            subset["true_label"],
            subset["predicted_clean"],
            labels=ALL_LABELS_WITH_ERROR,
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap_color,
            xticklabels=[
                "other_error" if label == "inne_blad" else label
                for label in ALL_LABELS_WITH_ERROR
            ],
            yticklabels=[
                "other_error" if label == "inne_blad" else label
                for label in ALL_LABELS_WITH_ERROR
            ],
            cbar=False,
        )

        plt.title(
            f"Confusion Matrix — {desc.title()} Configuration\n"
            f"{model_name} | {prompt_name}"
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                f"wykres_macierz_{desc}_{safe_filename(model_name)}.png",
            )
        )
        plt.close()

    # ---------------------------------------------------------
    # Table 8: Most frequent errors
    # ---------------------------------------------------------

    errors = best_subset[
        best_subset["true_label"]
        != best_subset["predicted_clean"]
    ]

    if not errors.empty:
        error_counts = (
            errors.groupby(["true_label", "predicted_clean"])
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .rename(columns={
                "true_label": "True_Label",
                "predicted_clean": "Predicted_Label",
            })
        )

        error_counts["Predicted_Label"] = (
            error_counts["Predicted_Label"]
            .replace({"inne_blad": "other_error"})
        )

        error_counts.to_csv(
            os.path.join(
                TABLES_DIR,
                "tabela_8_najczestsze_pomylki.csv",
            ),
            index=False,
        )

    # ---------------------------------------------------------
    # Chart 12: Macro precision heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 8))

    pivot_prec = metrics_df.pivot(
        index="Model",
        columns="Prompt",
        values="Macro_Precision",
    )

    sns.heatmap(
        pivot_prec,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5,
    )

    plt.title("Macro Precision Heatmap: Models vs. Prompts")
    plt.xlabel("Prompt Type")
    plt.ylabel("Model")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_12_heatmapa_precision.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 13: Macro recall heatmap
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 8))

    pivot_rec = metrics_df.pivot(
        index="Model",
        columns="Prompt",
        values="Macro_Recall",
    )

    sns.heatmap(
        pivot_rec,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        linewidths=0.5,
    )

    plt.title("Macro Recall Heatmap: Models vs. Prompts")
    plt.xlabel("Prompt Type")
    plt.ylabel("Model")

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_13_heatmapa_recall.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Chart 14: Precision vs. recall
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=plot_df,
        x="Macro_Precision",
        y="Macro_Recall",
        hue="Model family",
        style="Model type",
        s=200,
    )

    for _, row in metrics_df.iterrows():
        plt.annotate(
            f"{row['Model']}\n{row['Prompt']}",
            (row["Macro_Precision"], row["Macro_Recall"]),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.3,
        label="Precision = Recall",
    )

    plt.title("Precision vs. Recall Trade-off")
    plt.xlabel("Macro Precision")
    plt.ylabel("Macro Recall")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_14_precision_vs_recall.png",
        )
    )
    plt.close()

    # ---------------------------------------------------------
    # Table 9 and Chart 15: Worst configuration metrics
    # ---------------------------------------------------------

    worst_subset = df[
        (df["model"] == worst_model)
        & (df["prompt_type"] == worst_prompt)
    ]

    worst_report = classification_report(
        worst_subset["true_label"],
        worst_subset["predicted_clean"],
        labels=VALID_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    worst_class_rows = [
        {
            "Class": label,
            "Precision": worst_report[label]["precision"],
            "Recall": worst_report[label]["recall"],
            "F1_Score": worst_report[label]["f1-score"],
        }
        for label in VALID_CATEGORIES
    ]

    worst_class_df = pd.DataFrame(worst_class_rows)

    worst_class_df.to_csv(
        os.path.join(
            TABLES_DIR,
            "tabela_9_najgorszy_model_per_klasa.csv",
        ),
        index=False,
    )

    melted_worst = worst_class_df.melt(
        id_vars="Class",
        value_vars=["Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=melted_worst,
        x="Class",
        y="Value",
        hue="Metric",
    )

    plt.title(
        "Precision, Recall and F1-Score by Class — Worst Configuration\n"
        f"{worst_model} | {worst_prompt}"
    )
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "wykres_15_metryki_per_klasa_worst.png",
        )
    )
    plt.close()


if __name__ == "__main__":
    generate_master_thesis_analysis()