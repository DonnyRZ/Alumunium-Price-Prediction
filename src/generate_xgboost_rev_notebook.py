import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_NOTEBOOK = ROOT / "notebooks" / "XGBoost.ipynb"
OUT_NOTEBOOK = ROOT / "notebooks" / "XGBoost-Rev.ipynb"


def as_lines(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError("Expected block not found while rewriting notebook.")
    return source.replace(old, new, 1)


def rewrite_fold_cell(source: str) -> str:
    old = """st = build_price_frame(df)
folds = build_yearly_walk_forward_folds(st)

print("rows in modeling frame:", len(st))
print("folds:", len(folds))
pd.DataFrame(folds)
"""
    new = """st = build_price_frame(df)
folds = build_yearly_walk_forward_folds(st)

print("Ringkasan data modeling")
print("rows in modeling frame:", len(st))
print("jumlah fold walk-forward:", len(folds))

fold_table = pd.DataFrame(folds).rename(
    columns={
        "fold_name": "Fold",
        "fold_order": "Urutan",
        "train_start": "Train mulai",
        "train_end": "Train selesai",
        "valid_start": "Valid mulai",
        "valid_end": "Valid selesai",
        "test_start": "Test mulai",
        "test_end": "Test selesai",
        "rows_train_raw": "Baris train awal",
        "rows_valid_raw": "Baris valid awal",
        "rows_test_raw": "Baris test awal",
    }
)
display(fold_table)

if not fold_table.empty:
    print("Cara baca warna: biru = train, oranye = valid, hijau = test.")

    fold_plot = pd.DataFrame(folds).copy()
    for col in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]:
        fold_plot[col] = pd.to_datetime(fold_plot[col])

    fig, ax = plt.subplots(figsize=(15, max(4.5, len(fold_plot) * 1.2)))

    for idx, row in fold_plot.iterrows():
        ax.barh(
            idx,
            (row["train_end"] - row["train_start"]).days + 1,
            left=row["train_start"],
            height=0.22,
            color="#4c78a8",
            alpha=0.95,
            label="Train" if idx == 0 else "",
        )
        ax.barh(
            idx,
            (row["valid_end"] - row["valid_start"]).days + 1,
            left=row["valid_start"],
            height=0.22,
            color="#f58518",
            alpha=0.95,
            label="Valid" if idx == 0 else "",
        )
        ax.barh(
            idx,
            (row["test_end"] - row["test_start"]).days + 1,
            left=row["test_start"],
            height=0.22,
            color="#54a24b",
            alpha=0.95,
            label="Test" if idx == 0 else "",
        )

    ax.set_yticks(np.arange(len(fold_plot)))
    ax.set_yticklabels(fold_plot["fold_name"])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Tanggal")
    ax.set_title("Visual pembagian train / valid / test per fold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
"""
    return replace_once(source, old, new)


def rewrite_training_cell(source: str) -> str:
    source = replace_once(
        source,
        """    if not tuning_df.empty:
        display(tuning_df.head(15))
""",
        """    if not tuning_df.empty:
        tuning_overview = pd.DataFrame([
            {"Ringkasan tuning": "Trial selesai", "Nilai": int((tuning_df["state"] == "complete").sum())},
            {"Ringkasan tuning": "Trial feasible", "Nilai": int(len(feasible_df)) if not feasible_df.empty else 0},
            {"Ringkasan tuning": "Sumber parameter terpilih", "Nilai": selected_params_source},
            {"Ringkasan tuning": "Tau no-harm terpilih", "Nilai": round(float(NOHARM_TAU_MULT_USED), 4)},
            {"Ringkasan tuning": "Ambang regime terpilih", "Nilai": round(float(REGIME_VOL_Z_USED), 4)},
        ])
        print("Ringkasan tuning Optuna")
        display(tuning_overview)
""",
    )

    source = replace_once(
        source,
        """reg_df = pd.DataFrame(rows)
pred_df = pd.DataFrame(pred_rows)
skip_df = pd.DataFrame(skip_rows)

print("selected_params_source:", selected_params_source)
print("REG_PARAMS_USED:", REG_PARAMS_USED)
print("NOHARM_TAU_MULT_USED:", NOHARM_TAU_MULT_USED)
print("REGIME_VOL_Z_USED:", REGIME_VOL_Z_USED)
print("reg rows:", len(reg_df), "| pred rows:", len(pred_df), "| skip rows:", len(skip_df))
if not skip_df.empty:
    display(skip_df)

reg_df
""",
        """reg_df = pd.DataFrame(rows)
pred_df = pd.DataFrame(pred_rows)
skip_df = pd.DataFrame(skip_rows)

print("Ringkasan training dan evaluasi")
print("selected_params_source:", selected_params_source)
print("REG_PARAMS_USED:", REG_PARAMS_USED)
print("NOHARM_TAU_MULT_USED:", NOHARM_TAU_MULT_USED)
print("REGIME_VOL_Z_USED:", REGIME_VOL_Z_USED)
print("jumlah fold yang dievaluasi:", len(reg_df))
print("jumlah baris prediksi tersimpan:", len(pred_df))

fold_eval_view = reg_df[
    ["fold_name", "fold_order", "train_n", "valid_n", "test_n"]
].rename(
    columns={
        "fold_name": "Fold",
        "fold_order": "Urutan",
        "train_n": "Train setelah purge",
        "valid_n": "Valid setelah purge",
        "test_n": "Test setelah purge",
    }
)
display(fold_eval_view)

if not skip_df.empty:
    skip_view = skip_df.rename(
        columns={
            "fold_name": "Fold",
            "fold_order": "Urutan",
            "train_n_raw": "Train awal",
            "valid_n_raw": "Valid awal",
            "test_n_raw": "Test awal",
            "train_n": "Train setelah purge",
            "valid_n": "Valid setelah purge",
            "test_n": "Test setelah purge",
            "reason": "Alasan dilewati",
        }
    )
    print(f"{len(skip_view)} fold dilewati karena ukuran split terlalu kecil setelah purge.")
    display(skip_view)
else:
    print("Tidak ada fold yang dilewati setelah purge.")

baseline_catalog = pd.DataFrame([
    {"Baseline": "persistence_close", "Ide singkat": "Harga berikutnya dianggap sama dengan harga saat ini."},
    {"Baseline": "drift_mean_ret", "Ide singkat": "Harga bergerak mengikuti rata-rata return historis train."},
    {"Baseline": "rolling_ret5_scaled", "Ide singkat": "Harga mengikuti rata-rata return pendek terbaru."},
    {"Baseline": "rolling_ret10_scaled", "Ide singkat": "Harga mengikuti rata-rata return 10 periode terakhir."},
    {"Baseline": "repeat_last_5event_move", "Ide singkat": "Harga mengulang momentum perubahan 5 event terakhir."},
])

baseline_choice_view = reg_df[
    [
        "fold_name",
        "baseline_candidate_count",
        "baseline_valid_name",
        "baseline_valid_mae",
        "baseline_test_name",
        "baseline_test_mae",
    ]
].rename(
    columns={
        "fold_name": "Fold",
        "baseline_candidate_count": "Jumlah baseline diuji",
        "baseline_valid_name": "Baseline terbaik di valid",
        "baseline_valid_mae": "MAE baseline di valid",
        "baseline_test_name": "Baseline yang dipakai di test",
        "baseline_test_mae": "MAE baseline di test",
    }
)

baseline_count_view = (
    reg_df["baseline_test_name"]
    .value_counts()
    .rename_axis("Baseline terpilih di test")
    .reset_index(name="Jumlah fold")
)

print("Baseline yang dibandingkan")
display(baseline_catalog)

print("Baseline yang terpilih pada tiap fold")
display(baseline_choice_view)

print("Baseline yang paling sering terpilih")
display(baseline_count_view)
""",
    )

    return source


def main() -> None:
    notebook = json.loads(SRC_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cells[0]["source"] = as_lines(
        """
        # XGBoost-Rev

        Notebook ini menyajikan model **XGBoost untuk prediksi harga `Close(t+1)`** dengan output yang dibuat lebih mudah dipahami.

        - Fokus notebook ini adalah membantu membaca hasil model dengan cepat dan jelas.
        - Setelan model, split, tuning, dan gate sudah final di notebook ini.
        - Jalankan notebook dari atas agar seluruh output tampil lengkap.
        """
    )

    cells[1]["source"] = as_lines(
        """
        ## Cara membaca notebook ini

        - **Prediksi 1** = prediksi mentah / langsung dari model.
        - **Prediksi 2** = prediksi final setelah pengaman `no-harm + regime gate`. Ini adalah output utama.
        - **Baseline** = model pembanding non-XGBoost yang dipilih di data validasi lalu dipakai di data test.
        - **Delta MAE** = `MAE model - MAE baseline`. Nilai **negatif** berarti model lebih baik daripada baseline.
        - Rentang **`P10 / P50 / P90`** dipakai pada **Prediksi 2** untuk menunjukkan rentang prediksi final.
        """
    )

    cells[4]["source"] = as_lines(rewrite_fold_cell("".join(cells[4]["source"])))
    cells[6]["source"] = as_lines(rewrite_training_cell("".join(cells[6]["source"])))

    cells[7]["source"] = as_lines(
        """
        # Ringkasan hasil utama yang mudah dibaca

        if reg_df.empty:
            raise RuntimeError("No evaluated folds. Check split constraints.")

        STRICT_MIN_FOLDS = 3
        STRICT_MAE_EPS = 1e-4
        STRICT_MAX_SIGN_FLIP = 0.34

        reg_eval = reg_df.sort_values("fold_order").reset_index(drop=True).copy()
        reg_eval["win_valid_noharm_strict"] = reg_eval["delta_valid_mae_noharm"] <= -STRICT_MAE_EPS
        reg_eval["win_test_noharm_strict"] = reg_eval["delta_test_mae_noharm"] <= -STRICT_MAE_EPS
        reg_eval["win_test_corr_strict"] = reg_eval["delta_test_mae_corr"] <= -STRICT_MAE_EPS

        status_valid_noharm = np.where(reg_eval["win_valid_noharm_strict"], -1, 1)
        status_test_noharm = np.where(reg_eval["win_test_noharm_strict"], -1, 1)
        sign_flip_rate_noharm = float((((status_valid_noharm * status_test_noharm) == -1).mean()))

        folds_ok = bool(len(reg_eval) >= STRICT_MIN_FOLDS)
        all_test_folds_win_noharm = bool(reg_eval["win_test_noharm_strict"].all())
        all_valid_folds_win_noharm = bool(reg_eval["win_valid_noharm_strict"].all())
        robust_pass_strict_noharm = bool(
            folds_ok
            and all_test_folds_win_noharm
            and (sign_flip_rate_noharm <= STRICT_MAX_SIGN_FLIP)
        )

        summary = pd.DataFrame([
            {
                "folds": int(len(reg_eval)),
                "valid_win_rate_noharm_strict": float(reg_eval["win_valid_noharm_strict"].mean()),
                "test_win_rate_noharm_strict": float(reg_eval["win_test_noharm_strict"].mean()),
                "test_win_rate_corr_strict": float(reg_eval["win_test_corr_strict"].mean()),
                "mean_delta_valid_mae_noharm": float(reg_eval["delta_valid_mae_noharm"].mean()),
                "mean_delta_test_mae_noharm": float(reg_eval["delta_test_mae_noharm"].mean()),
                "worst_delta_test_mae_noharm": float(reg_eval["delta_test_mae_noharm"].max()),
                "mean_delta_test_mae_corr": float(reg_eval["delta_test_mae_corr"].mean()),
                "worst_delta_test_mae_corr": float(reg_eval["delta_test_mae_corr"].max()),
                "mean_baseline_test_mae": float(reg_eval["baseline_test_mae"].mean()),
                "mean_xgb_noharm_test_mae": float(reg_eval["xgb_noharm_test_mae"].mean()),
                "mean_xgb_corr_test_mae": float(reg_eval["xgb_corr_test_mae"].mean()),
                "mean_xgb_noharm_test_smape": float(reg_eval["xgb_noharm_test_smape"].mean()),
                "mean_xgb_noharm_test_dir_acc_nonzero": float(np.nanmean(reg_eval["xgb_noharm_test_dir_acc_nonzero"])),
                "mean_xgb_noharm_test_dir_nonzero_share": float(np.nanmean(reg_eval["xgb_noharm_test_dir_nonzero_share"])),
                "mean_noharm_test_cov80": float(reg_eval["noharm_test_cov80"].mean()),
                "mean_noharm_test_width80": float(reg_eval["noharm_test_width80"].mean()),
                "mean_noharm_tau_abs": float(reg_eval["noharm_tau_abs"].mean()),
                "mean_regime_active_share_valid": float(reg_eval["regime_active_share_valid"].mean()),
                "mean_regime_active_share_test": float(reg_eval["regime_active_share_test"].mean()),
                "mean_gate_applied_share_valid": float(reg_eval["gate_applied_share_valid"].mean()),
                "mean_gate_applied_share_test": float(reg_eval["gate_applied_share_test"].mean()),
            }
        ])

        robust = pd.DataFrame([
            {
                "folds": int(len(reg_eval)),
                "strict_min_folds": int(STRICT_MIN_FOLDS),
                "folds_meet_min": folds_ok,
                "strict_mae_eps": float(STRICT_MAE_EPS),
                "all_test_folds_win_noharm": all_test_folds_win_noharm,
                "all_valid_folds_win_noharm": all_valid_folds_win_noharm,
                "sign_flip_rate_noharm": sign_flip_rate_noharm,
                "strict_max_sign_flip": float(STRICT_MAX_SIGN_FLIP),
                "sign_flip_ok_noharm": bool(sign_flip_rate_noharm <= STRICT_MAX_SIGN_FLIP),
                "robust_pass_strict_noharm": robust_pass_strict_noharm,
            }
        ])

        def delta_sentence(delta_value, label):
            if delta_value < -STRICT_MAE_EPS:
                return f"{label} rata-rata lebih baik dari baseline sebesar {abs(delta_value):.4f} MAE."
            if delta_value > STRICT_MAE_EPS:
                return f"{label} rata-rata masih lebih buruk dari baseline sebesar {delta_value:.4f} MAE."
            return f"{label} rata-rata hampir sama dengan baseline."

        istilah_df = pd.DataFrame([
            {"Istilah": "Prediksi 1", "Arti singkat": "Prediksi mentah langsung dari model."},
            {"Istilah": "Prediksi 2", "Arti singkat": "Prediksi final setelah pengaman; ini output utama."},
            {"Istilah": "Baseline", "Arti singkat": "Pembanding non-XGBoost yang dipakai untuk evaluasi."},
            {"Istilah": "Delta MAE", "Arti singkat": "MAE model dikurangi MAE baseline. Negatif = lebih baik."},
        ])

        ringkasan_utama = pd.DataFrame([
            {
                "Pertanyaan": "Berapa fold yang benar-benar dievaluasi?",
                "Jawaban singkat": f"{len(reg_eval)} fold",
            },
            {
                "Pertanyaan": "Bagaimana baseline pada data test?",
                "Jawaban singkat": f"MAE rata-rata {reg_eval['baseline_test_mae'].mean():.4f}",
            },
            {
                "Pertanyaan": "Bagaimana Prediksi 1 pada data test?",
                "Jawaban singkat": f"MAE {reg_eval['xgb_corr_test_mae'].mean():.4f} | delta {reg_eval['delta_test_mae_corr'].mean():+.4f}",
            },
            {
                "Pertanyaan": "Bagaimana Prediksi 2 pada data test?",
                "Jawaban singkat": f"MAE {reg_eval['xgb_noharm_test_mae'].mean():.4f} | delta {reg_eval['delta_test_mae_noharm'].mean():+.4f}",
            },
            {
                "Pertanyaan": "Seberapa pas interval Prediksi 2?",
                "Jawaban singkat": f"Coverage 80% = {reg_eval['noharm_test_cov80'].mean():.2%}",
            },
            {
                "Pertanyaan": "Seberapa sering gate aktif di test?",
                "Jawaban singkat": f"{reg_eval['gate_applied_share_test'].mean():.2%} baris",
            },
        ])

        checklist_df = pd.DataFrame([
            {"Syarat": "Leakage lulus", "Status": "Ya" if leak_status == "PASS" else "Tidak"},
            {"Syarat": "Jumlah fold minimum terpenuhi", "Status": "Ya" if folds_ok else "Tidak"},
            {"Syarat": "Semua fold valid dimenangkan Prediksi 2", "Status": "Ya" if all_valid_folds_win_noharm else "Tidak"},
            {"Syarat": "Semua fold test dimenangkan Prediksi 2", "Status": "Ya" if all_test_folds_win_noharm else "Tidak"},
            {"Syarat": "Sign flip masih dalam batas aman", "Status": "Ya" if sign_flip_rate_noharm <= STRICT_MAX_SIGN_FLIP else "Tidak"},
            {"Syarat": "Robustness ketat lulus", "Status": "Ya" if robust_pass_strict_noharm else "Tidak"},
        ])

        ringkasan_per_fold = reg_eval[
            [
                "fold_name",
                "baseline_test_name",
                "baseline_test_mae",
                "xgb_corr_test_mae",
                "delta_test_mae_corr",
                "xgb_noharm_test_mae",
                "delta_test_mae_noharm",
                "noharm_test_cov80",
                "gate_applied_share_test",
            ]
        ].rename(
            columns={
                "fold_name": "Fold",
                "baseline_test_name": "Baseline di test",
                "baseline_test_mae": "MAE Baseline",
                "xgb_corr_test_mae": "MAE Prediksi 1",
                "delta_test_mae_corr": "Delta MAE Prediksi 1",
                "xgb_noharm_test_mae": "MAE Prediksi 2",
                "delta_test_mae_noharm": "Delta MAE Prediksi 2",
                "noharm_test_cov80": "Coverage 80% Prediksi 2",
                "gate_applied_share_test": "Gate aktif",
            }
        ).copy()

        ringkasan_per_fold["Pemenang test"] = np.select(
            [
                ringkasan_per_fold["Delta MAE Prediksi 2"] < 0,
                ringkasan_per_fold["Delta MAE Prediksi 1"] < 0,
            ],
            ["Prediksi 2", "Prediksi 1"],
            default="Baseline / Imbang",
        )

        kolom_angka = [
            "MAE Baseline",
            "MAE Prediksi 1",
            "Delta MAE Prediksi 1",
            "MAE Prediksi 2",
            "Delta MAE Prediksi 2",
            "Coverage 80% Prediksi 2",
            "Gate aktif",
        ]
        ringkasan_per_fold[kolom_angka] = ringkasan_per_fold[kolom_angka].round(4)

        insight_lines = [
            delta_sentence(float(reg_eval["delta_test_mae_corr"].mean()), "Prediksi 1"),
            delta_sentence(float(reg_eval["delta_test_mae_noharm"].mean()), "Prediksi 2"),
            f"Coverage interval Prediksi 2 ada di {reg_eval['noharm_test_cov80'].mean():.2%}; target utamanya sekitar 80%.",
            f"Gate aktif di {reg_eval['gate_applied_share_test'].mean():.2%} baris test, jadi output final cenderung cukup konservatif.",
            "Checklist robust harus dibaca sebagai syarat lolos model, bukan sekadar statistik tambahan.",
        ]

        print("Panduan istilah")
        display(istilah_df)

        print("Ringkasan utama hasil model")
        display(ringkasan_utama)

        print("Insight utama")
        for line in insight_lines:
            print("-", line)

        print("Checklist kelulusan model")
        display(checklist_df)

        print("Arti checklist secara high level")
        print("- Leakage harus lulus agar pemisahan data aman.")
        print("- Prediksi 2 harus konsisten mengalahkan atau minimal tidak kalah dari baseline di valid dan test.")
        print(f"- Batas aman sign flip adalah {STRICT_MAX_SIGN_FLIP:.2f}; makin kecil biasanya makin stabil.")
        print("- Jika robustness ketat tidak lulus, model belum cukup stabil untuk dianggap aman.")

        print("Ringkasan hasil per fold")
        display(ringkasan_per_fold)
        """
    )

    cells[8]["source"] = as_lines(
        """
        # Rincian delta MAE vs baseline, lalu visualisasi

        delta_detail = reg_eval[
            [
                "fold_name",
                "baseline_test_name",
                "baseline_test_mae",
                "delta_valid_mae_noharm",
                "delta_test_mae_corr",
                "delta_test_mae_noharm",
                "xgb_corr_test_mae",
                "xgb_noharm_test_mae",
            ]
        ].rename(
            columns={
                "fold_name": "Fold",
                "baseline_test_name": "Baseline terkunci",
                "baseline_test_mae": "MAE Baseline (test)",
                "delta_valid_mae_noharm": "Delta MAE Prediksi 2 vs Baseline (valid)",
                "delta_test_mae_corr": "Delta MAE Prediksi 1 vs Baseline (test)",
                "delta_test_mae_noharm": "Delta MAE Prediksi 2 vs Baseline (test)",
                "xgb_corr_test_mae": "MAE Prediksi 1 (test)",
                "xgb_noharm_test_mae": "MAE Prediksi 2 (test)",
            }
        ).copy()

        delta_detail["Interpretasi test"] = np.select(
            [
                delta_detail["Delta MAE Prediksi 2 vs Baseline (test)"] < 0,
                delta_detail["Delta MAE Prediksi 1 vs Baseline (test)"] < 0,
            ],
            [
                "Prediksi 2 lebih baik dari baseline",
                "Prediksi 1 lebih baik dari baseline",
            ],
            default="Baseline masih lebih baik / imbang",
        )

        kolom_delta = [
            "MAE Baseline (test)",
            "Delta MAE Prediksi 2 vs Baseline (valid)",
            "Delta MAE Prediksi 1 vs Baseline (test)",
            "Delta MAE Prediksi 2 vs Baseline (test)",
            "MAE Prediksi 1 (test)",
            "MAE Prediksi 2 (test)",
        ]
        delta_detail[kolom_delta] = delta_detail[kolom_delta].round(4)

        delta_rata2 = pd.DataFrame([
            {
                "Ringkasan": "Rata-rata delta MAE Prediksi 2 vs Baseline (valid)",
                "Nilai": round(float(reg_eval["delta_valid_mae_noharm"].mean()), 4),
            },
            {
                "Ringkasan": "Rata-rata delta MAE Prediksi 1 vs Baseline (test)",
                "Nilai": round(float(reg_eval["delta_test_mae_corr"].mean()), 4),
            },
            {
                "Ringkasan": "Rata-rata delta MAE Prediksi 2 vs Baseline (test)",
                "Nilai": round(float(reg_eval["delta_test_mae_noharm"].mean()), 4),
            },
            {
                "Ringkasan": "Delta MAE terburuk Prediksi 2 vs Baseline (test)",
                "Nilai": round(float(reg_eval["delta_test_mae_noharm"].max()), 4),
            },
        ])

        print("Detail delta MAE terhadap baseline")
        print("Catatan penting: delta MAE negatif berarti model lebih baik daripada baseline.")
        display(delta_rata2)
        display(delta_detail)

        x = np.arange(len(reg_eval))
        width = 0.36

        fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=False)

        axes[0].bar(
            x - width / 2,
            reg_eval["delta_valid_mae_noharm"],
            width=width,
            color="#1f77b4",
            alpha=0.9,
            label="Valid - Prediksi 2",
        )
        axes[0].bar(
            x + width / 2,
            reg_eval["delta_test_mae_noharm"],
            width=width,
            color="#ff7f0e",
            alpha=0.9,
            label="Test - Prediksi 2",
        )
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(reg_eval["fold_name"], rotation=25)
        axes[0].set_title("Prediksi 2 vs Baseline")
        axes[0].set_ylabel("Delta MAE (negatif = lebih baik)")
        axes[0].legend(loc="best")

        axes[1].bar(
            x - width / 2,
            reg_eval["delta_test_mae_corr"],
            width=width,
            color="#9467bd",
            alpha=0.9,
            label="Test - Prediksi 1",
        )
        axes[1].bar(
            x + width / 2,
            reg_eval["delta_test_mae_noharm"],
            width=width,
            color="#2ca02c",
            alpha=0.9,
            label="Test - Prediksi 2",
        )
        axes[1].axhline(0, color="black", lw=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(reg_eval["fold_name"], rotation=25)
        axes[1].set_title("Prediksi 1 vs Prediksi 2 pada Data Test")
        axes[1].set_ylabel("Delta MAE (negatif = lebih baik)")
        axes[1].legend(loc="best")

        plt.tight_layout()
        plt.show()
        """
    )

    cells[9]["source"] = as_lines(
        """
        # Visual: actual vs predicted price over time (dipisah antara Prediksi 1 dan Prediksi 2)

        pred_df["Date"] = pd.to_datetime(pred_df["Date"])

        def plot_prediction_stream(pred_col, label_pred, color, use_band=False):
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

            for ax, split in zip(axes, ["valid", "test"]):
                d = pred_df[pred_df["split"] == split].sort_values("Date")

                ax.plot(d["Date"], d["y_true_price_t5"], color="black", lw=1.3, label="Harga aktual (Close t+1)")
                ax.plot(d["Date"], d["baseline_price_t5"], color="#7f7f7f", lw=1.0, alpha=0.9, label="Baseline terkunci")

                if use_band:
                    ax.fill_between(
                        d["Date"],
                        d["y_pred_p10_t5"],
                        d["y_pred_p90_t5"],
                        color=color,
                        alpha=0.12,
                        label="Rentang Prediksi 2 (P10-P90)",
                    )

                ax.plot(d["Date"], d[pred_col], color=color, lw=1.2, alpha=0.95, label=label_pred)

                ax.set_title(f"{split.upper()} | Harga aktual vs {label_pred}")
                ax.set_ylabel("Price")
                ax.xaxis.set_major_locator(mdates.YearLocator(1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(axis="x", rotation=45)
                ax.legend(loc="upper left")

            plt.tight_layout()
            plt.show()

        plot_prediction_stream(
            pred_col="y_pred_corr_t5",
            label_pred="Prediksi 1 (mentah langsung dari model)",
            color="#ff7f0e",
            use_band=False,
        )

        plot_prediction_stream(
            pred_col="y_pred_p50_t5",
            label_pred="Prediksi 2 (final setelah pengaman)",
            color="#1f77b4",
            use_band=True,
        )
        """
    )

    cells[10]["source"] = as_lines(
        """
        # Perbandingan performa test yang paling penting

        te = pred_df[pred_df["split"] == "test"].copy()
        te["Date"] = pd.to_datetime(te["Date"])
        te = te.sort_values(["Date", "fold_order"]).reset_index(drop=True)

        te["resid_pred1"] = te["y_true_price_t5"] - te["y_pred_corr_t5"]
        te["resid_pred2"] = te["y_true_price_t5"] - te["y_pred_p50_t5"]
        te["abs_err_pred1"] = np.abs(te["resid_pred1"])
        te["abs_err_pred2"] = np.abs(te["resid_pred2"])
        te["abs_err_baseline"] = np.abs(te["y_true_price_t5"] - te["baseline_price_t5"])

        summary_diag = pd.DataFrame([
            {
                "Versi": "Baseline",
                "MAE": float(mean_absolute_error(te["y_true_price_t5"], te["baseline_price_t5"])),
                "Delta vs Baseline": 0.0,
                "Peran": "Pembanding utama",
            },
            {
                "Versi": "Prediksi 1",
                "MAE": float(mean_absolute_error(te["y_true_price_t5"], te["y_pred_corr_t5"])),
                "Delta vs Baseline": float(mean_absolute_error(te["y_true_price_t5"], te["y_pred_corr_t5"])) - float(mean_absolute_error(te["y_true_price_t5"], te["baseline_price_t5"])),
                "Peran": "Prediksi mentah",
            },
            {
                "Versi": "Prediksi 2",
                "MAE": float(mean_absolute_error(te["y_true_price_t5"], te["y_pred_p50_t5"])),
                "Delta vs Baseline": float(mean_absolute_error(te["y_true_price_t5"], te["y_pred_p50_t5"])) - float(mean_absolute_error(te["y_true_price_t5"], te["baseline_price_t5"])),
                "Peran": "Output final",
            },
        ])

        fold_diag = te.groupby("fold_name", as_index=False).agg(
            mae_baseline=("abs_err_baseline", "mean"),
            mae_pred1=("abs_err_pred1", "mean"),
            mae_pred2=("abs_err_pred2", "mean"),
            gate_applied_share=("gate_applied", "mean"),
        )

        fold_diag["winner"] = fold_diag[["mae_baseline", "mae_pred1", "mae_pred2"]].idxmin(axis=1).map(
            {
                "mae_baseline": "Baseline",
                "mae_pred1": "Prediksi 1",
                "mae_pred2": "Prediksi 2",
            }
        )

        fold_diag_view = fold_diag.rename(
            columns={
                "fold_name": "Fold",
                "mae_baseline": "MAE Baseline",
                "mae_pred1": "MAE Prediksi 1",
                "mae_pred2": "MAE Prediksi 2",
                "gate_applied_share": "Gate aktif",
                "winner": "Pemenang MAE",
            }
        ).copy()

        kolom_fold = ["MAE Baseline", "MAE Prediksi 1", "MAE Prediksi 2", "Gate aktif"]
        summary_diag[["MAE", "Delta vs Baseline"]] = summary_diag[["MAE", "Delta vs Baseline"]].round(4)
        fold_diag_view[kolom_fold] = fold_diag_view[kolom_fold].round(4)

        print("Ringkasan performa pada data test")
        display(summary_diag)

        print("Cara membaca bagian ini")
        print("-", "MAE adalah error utama; makin kecil makin baik.")
        print("-", "Delta vs Baseline negatif berarti versi tersebut lebih baik daripada baseline.")
        print("-", "Kolom pemenang menunjukkan siapa yang paling kecil error-nya di tiap fold.")
        print("-", "Gate aktif menunjukkan seberapa sering Prediksi 2 benar-benar dipakai sebagai output final.")

        print("Metrik per fold pada data test")
        display(fold_diag_view)

        x = np.arange(len(fold_diag))
        width = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))

        axes[0].bar(x - width, fold_diag["mae_baseline"], width=width, color="#7f7f7f", label="Baseline")
        axes[0].bar(x, fold_diag["mae_pred1"], width=width, color="#ff7f0e", label="Prediksi 1")
        axes[0].bar(x + width, fold_diag["mae_pred2"], width=width, color="#1f77b4", label="Prediksi 2")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(fold_diag["fold_name"], rotation=20)
        axes[0].set_title("Perbandingan MAE per Fold (Test)")
        axes[0].set_ylabel("MAE")
        axes[0].legend(loc="best")

        axes[1].bar(x, fold_diag["gate_applied_share"], width=0.4, color="#2ca02c")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(fold_diag["fold_name"], rotation=20)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Seberapa sering Gate Aktif")
        axes[1].set_ylabel("Proporsi baris test")

        plt.tight_layout()
        plt.show()
        """
    )

    cells[11]["source"] = as_lines(
        """
        # Keputusan akhir model

        leakage_pass = bool(leak_status == "PASS")

        robust_row = robust.iloc[0]
        folds_ok = bool(robust_row["folds_meet_min"])
        all_test_folds_win_noharm = bool(robust_row["all_test_folds_win_noharm"])
        all_valid_folds_win_noharm = bool(robust_row["all_valid_folds_win_noharm"])
        sign_flip_ok_noharm = bool(robust_row["sign_flip_ok_noharm"])
        robustness_pass_strict_noharm = bool(robust_row["robust_pass_strict_noharm"])

        hard_fail_reasons = []
        if not leakage_pass:
            hard_fail_reasons.append("leakage_fail")
        if not folds_ok:
            hard_fail_reasons.append("insufficient_folds")
        if not all_valid_folds_win_noharm:
            hard_fail_reasons.append("valid_not_outperform_locked_baseline")
        if not all_test_folds_win_noharm:
            hard_fail_reasons.append("noharm_model_loses_to_locked_baseline_on_some_test_folds")
        if not sign_flip_ok_noharm:
            hard_fail_reasons.append("noharm_sign_flip_rate_too_high")

        overall_go = bool(leakage_pass and robustness_pass_strict_noharm)

        optuna_feasible_trial_count = 0
        if "tuning_df" in globals() and isinstance(tuning_df, pd.DataFrame) and not tuning_df.empty:
            if {"state", "fail_count_valid", "mean_delta_valid_noharm"}.issubset(set(tuning_df.columns)):
                optuna_feasible_trial_count = int(
                    (
                        (tuning_df["state"] == "complete")
                        & (tuning_df["fail_count_valid"] <= 0)
                        & (tuning_df["mean_delta_valid_noharm"] <= -MUST_WIN_VALID_EPS)
                    ).sum()
                )

        decision = {
            "target": f"close_t{H}",
            "model_type": "xgboost_residual_correction_with_noharm_and_regime_gate",
            "primary_gate_stream": "xgb_noharm_price_p50",
            "output_contract": [f"price_p10_t{H}", f"price_p50_t{H}", f"price_p90_t{H}"],
            "leakage_pass": leakage_pass,
            "robustness_pass_strict_noharm": robustness_pass_strict_noharm,
            "folds_meet_min": folds_ok,
            "all_test_folds_win_noharm": all_test_folds_win_noharm,
            "all_valid_folds_win_noharm": all_valid_folds_win_noharm,
            "sign_flip_ok_noharm": sign_flip_ok_noharm,
            "overall_go_price_model": overall_go,
            "hard_fail_reasons": hard_fail_reasons,
            "mean_delta_test_mae_noharm": float(reg_eval["delta_test_mae_noharm"].mean()),
            "worst_delta_test_mae_noharm": float(reg_eval["delta_test_mae_noharm"].max()),
            "mean_delta_test_mae_corr": float(reg_eval["delta_test_mae_corr"].mean()),
            "mean_baseline_test_mae": float(reg_eval["baseline_test_mae"].mean()),
            "mean_xgb_noharm_test_mae": float(reg_eval["xgb_noharm_test_mae"].mean()),
            "mean_xgb_corr_test_mae": float(reg_eval["xgb_corr_test_mae"].mean()),
            "mean_xgb_noharm_test_smape": float(reg_eval["xgb_noharm_test_smape"].mean()),
            "mean_xgb_noharm_test_dir_acc_nonzero": float(np.nanmean(reg_eval["xgb_noharm_test_dir_acc_nonzero"])),
            "mean_xgb_noharm_test_dir_nonzero_share": float(np.nanmean(reg_eval["xgb_noharm_test_dir_nonzero_share"])),
            "mean_noharm_test_cov80": float(reg_eval["noharm_test_cov80"].mean()),
            "mean_regime_active_share_test": float(reg_eval["regime_active_share_test"].mean()),
            "mean_gate_applied_share_test": float(reg_eval["gate_applied_share_test"].mean()),
            "feature_count": int(len(FEATURES)),
            "selected_params_source": selected_params_source,
            "optuna_feasible_trial_count": int(optuna_feasible_trial_count),
            "reg_params_used": REG_PARAMS_USED,
            "noharm_tau_mult_used": float(NOHARM_TAU_MULT_USED),
            "noharm_tau_mult_min": float(NOHARM_TAU_MULT_MIN),
            "regime_vol_z_used": float(REGIME_VOL_Z_USED),
            "regime_active_target_low": float(REGIME_ACTIVE_TARGET_LOW),
            "regime_active_target_high": float(REGIME_ACTIVE_TARGET_HIGH),
            "use_optuna": bool(USE_OPTUNA),
            "optuna_trials": int(OPTUNA_N_TRIALS) if USE_OPTUNA else 0,
        }

        decision_view = pd.DataFrame([
            {"Poin": "Model layak dipakai?", "Nilai": "Ya" if overall_go else "Tidak"},
            {"Poin": "Output utama", "Nilai": "Prediksi 2"},
            {"Poin": "Leakage lulus?", "Nilai": "Ya" if leakage_pass else "Tidak"},
            {"Poin": "Semua fold valid dimenangkan Prediksi 2?", "Nilai": "Ya" if all_valid_folds_win_noharm else "Tidak"},
            {"Poin": "Semua fold test dimenangkan Prediksi 2?", "Nilai": "Ya" if all_test_folds_win_noharm else "Tidak"},
            {"Poin": "Alasan gagal utama", "Nilai": ", ".join(hard_fail_reasons) if hard_fail_reasons else "Tidak ada"},
            {"Poin": "Sumber parameter", "Nilai": selected_params_source},
        ])

        params_view = pd.DataFrame(
            [{"Parameter": key, "Nilai": value} for key, value in REG_PARAMS_USED.items()]
        )

        print("Keputusan akhir model")
        display(decision_view)

        print("Cara membaca keputusan")
        print("- Jika model belum layak dipakai, lihat kolom alasan gagal utama terlebih dahulu.")
        print("- Output utama yang dipakai tetap Prediksi 2, bukan Prediksi 1.")
        print("- Sumber parameter membantu melihat apakah model memakai default atau hasil tuning.")

        print("Parameter XGBoost yang dipakai")
        display(params_view)
        """
    )

    cells[12]["source"] = as_lines(
        """
        # Save artifacts
        SAVE_ARTIFACTS = False
        prefix = "xgb_v6_price_t5_rev"

        if SAVE_ARTIFACTS:
            reg_eval.to_csv(OUT_DIR / f"{prefix}_fold_results.csv", index=False)
            pred_df.to_csv(OUT_DIR / f"{prefix}_predictions.csv", index=False)
            audit_df.to_csv(OUT_DIR / f"{prefix}_leakage_audit.csv", index=False)
            viol_df.to_csv(OUT_DIR / f"{prefix}_leakage_violations.csv", index=False)
            summary.to_csv(OUT_DIR / f"{prefix}_summary.csv", index=False)
            robust.to_csv(OUT_DIR / f"{prefix}_robustness.csv", index=False)
            if "tuning_df" in globals() and isinstance(tuning_df, pd.DataFrame) and not tuning_df.empty:
                tuning_df.to_csv(OUT_DIR / f"{prefix}_optuna_trials.csv", index=False)
            (OUT_DIR / f"{prefix}_decision.json").write_text(json.dumps(decision, indent=2))
            print("saved artifacts with prefix:", prefix)
        else:
            print("SAVE_ARTIFACTS=False -> artifacts not written")
        """
    )

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    OUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    print(f"Notebook written to {OUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
