#!/bin/bash

# This PLOT script documents the exact procedures we use to make the plots


_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly DATASET_PATH=${_DIR}/../data
readonly RESULTS_PATH=${_DIR}/../_results
readonly PROJECTS=(
      "asterisk-java_asterisk-java"
      "Bukkit_Bukkit"
      "apache_commons-configuration"
      "apache_commons-csv"
      "apache_commons-lang"
      "apache_commons-net"
	    "apache_commons-validator"
	    "frizbog_gedcom4j"
	    "mikera_vectorz"
	    "zeroturnaround_zt-exec"
	    )

readonly MODELS=(
        "Fail-Basic"
        "Fail-Code"
        "Fail-ABS"
        "Ekstazi-Basic"
        "Ekstazi-Code"
        "Ekstazi-ABS"
        "BM25Baseline"
        "Fail-Basic-BM25Baseline"
        "Ekstazi-Basic-BM25Baseline"
        "randomforest"
        "boosting"
)

# Get line plot for all the projects
function plot_recall_selection_rate_all() {
    for proj in "${PROJECTS[@]}"; do
      plot_perfect_select_subset_recall_selection_rate $proj
      plot_rank_models_recall_selection_rate $proj
    done
    # Handle corner case: commons-csv, zt-exec
    for PROJ in apache_commons-csv zeroturnaround_zt-exec; do
        python -m pts.main make_plots --which=rank-model-plot-data --data_type=Fail-Code --data_type=Fail-Basic\
               --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
               --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting\
               --project=$PROJ
        python -m pts.main make_plots --which=plot-recall-selection-rate --data_type=Fail-Code --data_type=Fail-Basic\
               --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
               --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --data_type=perfect\
               --project=$PROJ
    done     
}

function plot_first_failure_recall_selection_rate_all() {
    for proj in "${PROJECTS[@]}"; do
	plot_first_failure_recall_selection_rate $proj
    done
}

function plot_first_failure_tool_all() {
    for proj in "${PROJECTS[@]}"; do
	plot_first_failure_recall_starts_selection_rate $proj
	plot_first_failure_recall_ekstazi_selection_rate $proj
    done
}

function plot_select_subset_tool_all() {
    for proj in "${PROJECTS[@]}"; do
      plot_rank_models_starts_select_subset_recall_selection_rate $proj
      plot_rank_models_ekstazi_select_subset_recall_selection_rate $proj
    done
    for PROJ in apache_commons-csv; do
        # python -m pts.main make_plots --which=subset-recall-selection-data --data_type=Fail-Code --data_type=Fail-Basic\
	#          --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	#          --data_type=Ekstazi-ABS --data_type=BM25Baseline \
    	#      --project=$PROJ --subset=STARTS
        python -m pts.main make_plots --which=plot-subset-recall-selection --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-ABS --data_type=BM25Baseline \
               --project=$PROJ  --subset=STARTS
        # python -m pts.main make_plots --which=subset-recall-selection-data --data_type=Fail-Code --data_type=Fail-Basic\
	#          --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	#          --data_type=Ekstazi-ABS --data_type=BM25Baseline \
    	#      --project=$PROJ --subset=Ekstazi

        python -m pts.main make_plots --which=plot-subset-recall-selection --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-ABS --data_type=BM25Baseline \
	        --project=$PROJ  --subset=Ekstazi
    done

}

function boxplot_best_selection_rate_all() {
    for proj in "${PROJECTS[@]}"; do
      plot_best_select_rate_box_plot $proj
    done
    for PROJ in apache_commons-csv zeroturnaround_zt-exec; do
        python -m pts.main make_plots --which="best-select-rate-boxplot" --project=$PROJ\
               --data_type=Fail-Basic --data_type=Fail-Code\
               --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
	             --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting
    done
}


function tool_perfect_recall_selection_rate_data() {
    for proj in "${PROJECTS[@]}"; do
	      python -m pts.main make_plots --which=tools-perfect-data --project=$proj --tool=STARTS
	      python -m pts.main make_plots --which=tools-perfect-data --project=$proj --tool=Ekstazi
    done
}

function boxplot_best_selection_rate_for_types() {
    for proj in "${PROJECTS[@]}"; do
	plot_best_select_rate_box_plot_for_types $proj
    done
}

function plot_roc_curve_proj_all() {
    for proj in "${PROJECTS[@]}"; do
	plot_roc_curve $proj
    done
}

function plot_pr_curve_proj_all() {
    for proj in "${PROJECTS[@]}"; do
	plot_pr_curve $proj
    done
}

function plot_roc_curve() {
        python -m pts.main make_plots --which=ROC-curve-data --data_type=Ekstazi-Basic\
    	     --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
    	     --data_type=Fail-ABS --project="$1"
    	  python -m pts.main make_plots  --which=plot-roc-curve --data_type=Ekstazi-Basic\
    	     --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
    	     --data_type=Fail-ABS --project="$1"
}

function plot_pr_curve() {
        python -m pts.main make_plots --which=pr-curve-data --data_type=Ekstazi-Basic\
    	     --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
    	     --data_type=Fail-ABS --project="$1"
    	  python -m pts.main make_plots --which=plot-pr-curve --data_type=Ekstazi-Basic\
    	     --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
    	     --data_type=Fail-ABS --project="$1"
}

# Get the data for plotting test-recall V.S. selection rate curve plot for all the baseline models and make the plot
function plot_rank_models_recall_selection_rate() {
    python -m pts.main make_plots --which=rank-model-plot-data --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --data_type=xgboost --data_type=randomforest\
     	     --project="$1"
    python -m pts.main make_plots --which=plot-recall-selection-rate --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --data_type=perfect --data_type=xgboost --data_type=randomforest\
	         --project="$1"
}


# Get the data for plotting test-recall V.S. selection rate curve plot for all the baseline models and make the plot
# Here the selection rate is # test selected / # test selected by tools
function plot_rank_models_starts_select_subset_recall_selection_rate() {

    # python -m pts.main make_plots --which=subset-recall-selection-data --data_type=Fail-Code --data_type=Fail-Basic\
    # 	         --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
    # 	         --data_type=Ekstazi-ABS --data_type=BM25Baseline --data_type=randomforest\
    # 	     --project="$1" --subset=STARTS
    python -m pts.main make_plots --which=plot-subset-recall-selection --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-ABS --data_type=BM25Baseline --data_type=randomforest\
	         --project="$1"  --subset=STARTS
}

function plot_rank_models_ekstazi_select_subset_recall_selection_rate() {
    	  # python -m pts.main make_plots --which=subset-recall-selection-data --data_type=Fail-Code --data_type=Fail-Basic\
	  #        --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	  #        --data_type=Ekstazi-ABS --data_type=BM25Baseline\
    	  #    --project="$1" --subset=Ekstazi
        python -m pts.main make_plots --which=plot-subset-recall-selection --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=Ekstazi-Code --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-ABS --data_type=BM25Baseline\
	        --project="$1"  --subset=Ekstazi
}


# Make the plots for finding the first failure on the tool subset.
function plot_perfect_select_subset_recall_selection_rate() {
        python -m pts.main make_plots --which=perfect-subset-recall-selection-data --project="$1" --subset=All
        python -m pts.main make_plots --which=perfect-subset-recall-selection-data --project="$1" --subset=STARTS
        python -m pts.main make_plots --which=perfect-subset-recall-selection-data --project="$1" --subset=Ekstazi
}

function plot_first_failure_recall_selection_rate() {
      python -m pts.main make_plots --which=first-failure-recall-selection-data --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=All
      python -m pts.main make_plots --which=plot-first-failure-recall-selection --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=All
}

function plot_first_failure_recall_ekstazi_selection_rate() {
      python -m pts.main make_plots --which=first-failure-recall-selection-data --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=Ekstazi
      python -m pts.main make_plots --which=plot-first-failure-recall-selection --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=Ekstazi
}

function plot_first_failure_recall_starts_selection_rate() {
      python -m pts.main make_plots --which=first-failure-recall-selection-data --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=STARTS
      python -m pts.main make_plots --which=plot-first-failure-recall-selection --project="$1" --data_type=Ekstazi-Basic\
	         --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Code --data_type=Fail-Basic\
	         --data_type=Fail-ABS --data_type=TFIDFBaseline --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	         --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --subset=STARTS
}

function plot_tools_all() {
    plot_seq2pred_model_tools apache_commons-dbcp
    plot_seq2pred_model_tools apache_commons-codec
    plot_seq2pred_model_tools logstash_logstash-logback-encoder
}

# Plot the barplot for each sha for each model to show the best selection-rate to get 1.0 test-Recall
function plot_rank_models_barplot() {
        python -m pts.main make_plots --which=rank-models-barplot --project="$1"\
               --data_type=STARTS-Basic\
               --data_type=STARTS-Code --data_type=STARTS-ABS --data_type=Fail-Basic --data_type=Fail-Code\
               --data_type=Fail-ABS
}

function plot_concise_rank_models_barplot() {
        python -m pts.main make_plots --which=rank-models-barplot --project="$1"\
               --data_type=Fail-Basic
}


function plot_rank_barplot() {
        python -m pts.main make_plots --which=rank-model-test-rank-barplot --project="$1"\
               --data_type=STARTS-Basic\
               --data_type=STARTS-Code --data_type=STARTS-ABS --data_type=Fail-Basic --data_type=Fail-Code\
               --data_type=Fail-ABS
}

function plot_best_select_rate_box_plot(){
        python -m pts.main make_plots --which="best-select-rate-boxplot" --project="$1"\
               --data_type=Ekstazi-Basic\
               --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Basic --data_type=Fail-Code\
               --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Basic-BM25Baseline\
               --data_type=randomforest --data_type=xgboost \
	             --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting
}

function plot_best_select_rate_box_plot_for_types(){
        # types: "simple-rule", "not-simple-rule", "newly-added-tests", "not-newly-added-tests", "killed-tests", "not-killed-tests"
        python -m pts.main make_plots --which="best-select-rate-boxplot-for-types" --project="$1"\
               --data_type=Ekstazi-Basic\
               --data_type=Ekstazi-Code --data_type=Ekstazi-ABS --data_type=Fail-Basic --data_type=Fail-Code\
               --data_type=Fail-ABS --data_type=BM25Baseline --data_type=Fail-Code-BM25Baseline\
	       --data_type=Ekstazi-Basic-BM25Baseline --data_type=boosting --types=simple-rule --types=not-simple-rule\
	       --types=newly-added-tests --types=not-newly-added-tests --types=killed-tests --types=not-killed-tests\
	       --types=no-simple-rule-newly-added-tests --types=no-simple-rule-not-newly-added-tests --types=no-simple-rule-killed-tests\
	       --types=no-simple-rule-not-killed-tests
}

function num_of_test_cases_raw_eval_data_lineplot() {
   # shellcheck disable=SC2145
   python -m pts.main make_plots --which=num_of_test_cases_raw_eval_data_lineplot  --projects="${PROJECTS[*]}"\
          --search_span=10 --search_span=20
}

function num_of_test_cases_mutated_eval_data_lineplot() {
   python -m pts.main make_plots --which=num_of_test_cases_mutated_eval_data_lineplot --project="$1"
}

function num_of_test_cases_mutated_eval_data_all() {
    for proj in "${PROJECTS[@]}"; do
        num_of_test_cases_mutated_eval_data_lineplot $proj &
    done
}

function recall_vs_selection_boxplot_plots_layout(){
    python -m pts.main make_plots --which=recall-vs-selection-boxplot-plots-layout  --projects="${PROJECTS[*]}"
}

function number_of_changed_files_vs_select_rate_plots_layout(){
    python -m pts.main make_plots --which=number-of-changed-files-vs-select-rate-plots-layout  --projects="${PROJECTS[*]}"
}

function num_changed_files_vs_avg_selection_rate_barplot(){
    python -m pts.main make_plots --which=num-changed-files-vs-avg-selection-rate-barplot --projects="${PROJECTS[*]}" --models="${MODELS[*]}" --type="simple-rule-average"
    python -m pts.main make_plots --which=num-changed-files-vs-avg-selection-rate-barplot --projects="${PROJECTS[*]}" --models="${MODELS[*]}" --type="simple-rule-safe"
        python -m pts.main make_plots --which=num-changed-files-vs-avg-selection-rate-barplot --projects="${PROJECTS[*]}" --models="${MODELS[*]}" --type="no-simple-rule-average"
    python -m pts.main make_plots --which=num-changed-files-vs-avg-selection-rate-barplot --projects="${PROJECTS[*]}" --models="${MODELS[*]}" --type="no-simple-rule-safe"
}

function boxplot_end_to_end_time(){
  python -m pts.main make_plots --which=boxplot-end-to-end-time --projects="${PROJECTS[*]}" --subset=Ekstazi --models="${MODELS[*]}"
  python -m pts.main make_plots --which=boxplot-end-to-end-time --projects="${PROJECTS[*]}" --subset=STARTS --models="${MODELS[*]}"
}

function boxplot_selection_time(){
  python -m pts.main make_plots --which=boxplot-selection-time --projects="${PROJECTS[*]}" --subset=Ekstazi --models="${MODELS[*]}"
  python -m pts.main make_plots --which=boxplot-selection-time --projects="${PROJECTS[*]}" --subset=STARTS --models="${MODELS[*]}"
}
# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"


# ==========
# Some notes of useful Bash commands

# Export Anaconda environment
# conda env export --from-history > env.yml

# Load Anaconda envrionment
# conda env create -n NAME -f env.yml
