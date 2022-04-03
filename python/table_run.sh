#!/bin/bash

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

readonly SUBSETS=(
        "All"
        "Ekstazi"
        "STARTS"
)

readonly DATATYPES=(
        "Fail-Basic"
        "Fail-Code"
        "Fail-ABS"
        "Ekstazi-Basic"
        "Ekstazi-Code"
        "Ekstazi-ABS"
        "BM25Baseline"
        "randomforest"
)

readonly RTSTOOLS=(
        "Ekstazi"
        "STARTS"
)

#############
# Scripts for the tables in the paper
#######################
function make_real_failure_results_table() {
        python -m pts.main make_tables --which=real-failed-test-no-rule-stats
        python -m pts.main make_tables --which=real-failure-results-paper-table
}

function make_avg_execution_time_table() {
        python -m pts.main make_tables --which=subset-execution-time --subset=Ekstazi --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}"
        python -m pts.main make_tables --which=subset-execution-time --subset=STARTS --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}"
}

function make_avg_end_to_end_time_table() {
        python -m pts.main make_tables --which=subset-end-to-end-time --subset=Ekstazi --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}"
        python -m pts.main make_tables --which=subset-end-to-end-time --subset=STARTS --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}"
}

function make_avg_end_to_end_time_table_change_selection_rate() {
        python -m pts.main make_tables --which=subset-end-to-end-time --subset=Ekstazi --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}" --change-selection-rate="True"
        python -m pts.main make_tables --which=subset-end-to-end-time --subset=STARTS --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}" --change-selection-rate="True"
}

function safe_selection_rate_table() {
        python -m pts.main make_tables --which=cmp-best-safe-select-rate --projects="${PROJECTS[*]}"
}

function make_macros_used_in_eval() {
        python -m pts.main collect_metrics --which=stats-reported-in-eval
}

#########################################
function average_safe_selection_rate_table() {
        python -m pts.main make_tables --which=cmp-avg-safe-select-rate --projects="${PROJECTS[*]}"
}

function auc_table() {
        python -m pts.main make_tables --which=AUC-results-table-paper --projects="${PROJECTS[*]}" --subset=All
        python -m pts.main make_tables --which=AUC-results-table-paper --projects="${PROJECTS[*]}" --subset=Ekstazi
        python -m pts.main make_tables --which=AUC-results-table-paper --projects="${PROJECTS[*]}" --subset=STARTS
}

####################

function make_pit_mutant_table_all() {
        for proj in "${PROJECTS[@]}"; do
                project_pit_mutants_table $proj
        done
}

function collect_and_make_eval_results_table() {
        for proj in "${PROJECTS[@]}"; do
                # rank_model_eval_results_collect $proj
                ensemble_model_eval_results_collect $proj
                # rank_models_eval_results_table $proj
                # line_mapping_baselines_table $proj
                # boosting_model_eval_results_collect $proj
        done
}

function collect_ensemble_model_results_all() {
        for proj in "${PROJECTS[@]}"; do
                ensemble_model_eval_results_collect $proj
        done
}

function collect_confusion_matrix_all() {
        for proj in "${PROJECTS[@]}"; do
                collect_confusion_matrices $proj
        done
}

function collect_correct_confusion_matrix_all() {
        for proj in "${PROJECTS[@]}"; do
                collect_correct_confusion_matrices $proj
        done
}

function collect_test_selection_metrics_all() {
        for proj in "${PROJECTS[@]}"; do
                collect_test_selection_metrics $proj
        done
}

function collect_and_make_newly_added_tests_table() {
        python -m pts.main make_tables --which=pct-newly-added-tests --projects="${PROJECTS[*]}"
}

function collect_and_make_eval_data_table() {
        for proj in "${PROJECTS[@]}"; do
                collect_mutated_eval_data_metrics $proj
                mutated_eval_dataset_table $proj
        done
}

function make_ir_macros_all() {
        for proj in "${PROJECTS[@]}"; do
                rank_model_IR_baseline_table $proj
        done
}

function collect_ir_metrics_all() {
        for proj in "${PROJECTS[@]}"; do
                rank_model_eval_results_IR_baseline_collect $proj
        done
}

function collect_ealrts_metrics_all() {
        for proj in "${PROJECTS[@]}"; do
                rank_model_eval_results_EALRTS_baseline_collect $proj
        done
}

function collect_subset_select_rate_all() {
        for proj in "${PROJECTS[@]}"; do
                collect_best_safe_selection_rate $proj
        done
}

function make_auc_numbers_all() {
        for proj in "${PROJECTS[@]}"; do
                make_auc_score_numbers $proj
        done
}

function make_confusion_matrices_all() {
        for proj in "${PROJECTS[@]}"; do
                confusion_matrix_table $proj
                correct_confusion_matrix_table $proj
        done
}

#-------Collect METRICS FOR THE TABLES -------------------------------

function raw_dataset_metrics_collect() {
        python -m pts.main collect_metrics --which=raw-dataset
}

function collect_mutated_eval_data_metrics() {
        python -m pts.main collect_metrics --which=mutated-eval-data --project="$1"
}

function rank_model_eval_results_triplet_collect() {
        python -m pts.main collect_metrics --which=triplet-model-results --project="$1"
        python -m pts.main make_tables --which=triplet-model-numbers --project="$1"
}

function ensemble_model_eval_results_collect() {
       python -m pts.main collect_metrics --which=ensemble-model-results --project="$1"\
              --models=Fail-Code --models=BM25Baseline
       python -m pts.main make_tables --which=ensemble-model-numbers --model=Fail-Code-BM25Baseline\
              --project="$1"
       python -m pts.main collect_metrics --which=ensemble-model-results --project="$1"\
               	   --models=Ekstazi-Basic --models=BM25Baseline
       python -m pts.main make_tables --which=ensemble-model-numbers --model=Ekstazi-Basic-BM25Baseline\
                      --project="$1"
        python -m pts.main collect_metrics --which=ensemble-model-results --project="$1"\
               --models=Fail-Basic --models=BM25Baseline
        python -m pts.main make_tables --which=ensemble-model-numbers --model=Fail-Basic-BM25Baseline\
               --project="$1"
}

function boosting_model_eval_results_collect() {
        python -m pts.main collect_metrics --which=boosting-eval-results-collect --project="$1"
        python -m pts.main make_tables --which=boosting-model-numbers --project="$1"
}

function rank_model_eval_results_collect() {
        python -m pts.main collect_metrics --which=rank-model-eval-results  --project="$1"
}

function rank_model_eval_results_IR_baseline_collect() {
        python -m pts.main collect_metrics --which=rank-model-eval-results-IR-baseline  --project="$1"
}

function rank_model_eval_results_EALRTS_baseline_collect() {
        python -m pts.main collect_metrics --which=rank-model-eval-results-EALRTS-baseline  --project="$1"
}

function model_dataset_metrics_collect() {
        python -m pts.main collect_metrics --which=model-data
}

function pit_mutants_stats() {
        python -m pts.main collect_metrics --which=pit-mutants\
               --projects="$1"
}

function collect_data_metrics_rank_model() {
        python -m pts.main collect_metrics --which=proj-mutations --project="$1"
}

function collect_best_safe_selection_rate() {
        python -m pts.main collect_metrics --which=subset-selection-rate --project="$1"
}

function move_rank_results() {
        python -m pts.main collect_metrics --which=move-rank-models-results --models=boosting --projects="${PROJECTS[*]}"
}

function collect_confusion_matrices() {
        python -m pts.main collect_metrics --which=confusion-matrices --project="$1"
}

function collect_correct_confusion_matrices() {
        python -m pts.main collect_metrics --which=correct-confusion-matrices --project="$1"
}

# Get metrics for example, missed failed tests, pct newly added tests
function collect_test_selection_metrics() {
        python -m pts.main collect_metrics --which=test-selection-metrics\
               --project="$1"
}

function collect_avg_best_safe_selection_rate() {
        python -m pts.main collect_metrics --which=average-metric-across-projects --projects=asterisk-java_asterisk-java\
               --projects=Bukkit_Bukkit --projects=apache_commons-configuration --projects=apache_commons-csv\
               --projects=apache_commons-lang --projects=apache_commons-net --projects=apache_commons-validator\
               --projects=frizbog_gedcom4j --projects=mikera_vectorz --projects=zeroturnaround_zt-exec
}

#-------------TABLES MAKERS------------------------

# The first positional arg is to specify which project to collect.
function project_pit_mutants_table() {
        python -m pts.main make_tables --which=project-pit-mutants\
               --project="$1"
}

function avg_safe_selection_rate_table() {
        python -m pts.main make_tables --which=avg-best-safe-select-rate
}

function tools_no_deps_table() {
        python -m pts.main make_tables --which=rank_model_eval_results_with_no_deps_update --project="$1"
}

function rank_model_IR_baseline_table() {
        python -m pts.main make_tables --which=rank_model_IR_baseline_eval_results --project="$1"
}

function mutated_eval_dataset_table() {
        python -m pts.main make_tables --which=mutated-eval-dataset-stats --project="$1"
}

function raw_dataset_stats_table() {
        python -m pts.main make_tables --which=dataset-raw-stats
}

function model_data_stats_table() {
        python -m pts.main make_tables --which=dataset-model-stats
}

function rank_model_results_table() {
        python -m pts.main make_tables --which=rank-model-per-file-results --project="$1"
}

function rank_models_eval_results_table() {
        python -m pts.main make_tables --which=rank-model-eval-results --project="$1"
}

function line_mapping_baselines_table() {
        python -m pts.main make_tables --which=mutant-line-mapping-baseline-stats --project="$1"\
               --search_span=10 --search_span=20
}

function make_dataset_table() {
        python -m pts.main make_tables --which=paper-dataset-table --projects="${PROJECTS[*]}"
}

function perfect_best_safe_selection_rate_table() {
        # shellcheck disable=SC2145
        python -m pts.main make_tables --which=perfect-best-safe-selection-rate --projects="${PROJECTS[*]}"
}

function confusion_matrix_table() {
        python -m pts.main make_tables --which=confusion-matrices-table --project="$1"
}

function correct_confusion_matrix_table() {
        python -m pts.main make_tables --which=correct-confusion-matrices-table --project="$1"
}

# Make numbers for auc
function make_auc_score_numbers() {
        python -m pts.main make_tables --which=auc-score-number --project="$1"
}

function make_eval_on_real_failed_tests() {
        python -m pts.main make_tables --which=real-failed-test-stats
        python -m pts.main make_tables --which=real-failed-test-table
}

function make_eval_on_real_failed_tests_no_rule() {
        python -m pts.main make_tables --which=real-failed-test-no-rule-stats
        python -m pts.main make_tables --which=real-failed-test-no-rule-table
}

function make_auc_recall_selection() {
        # newly added EALRTS models in ${DATATYPES[*]} may cause error
        python -m pts.main make_tables --which=auc-recall-selection-stats --projects="${PROJECTS[*]}" --subsets="${SUBSETS[*]}" --datatypes="${DATATYPES[*]}"
        python -m pts.main make_tables --which=auc-recall-selection-table --projects="${PROJECTS[*]}" --subsets="${SUBSETS[*]}" --datatypes="${DATATYPES[*]}"
}

function make_auc_recall_first_failure_selection() {
        # newly added EALRTS models in ${DATATYPES[*]} may cause error
        python -m pts.main make_tables --which=first-failure-auc-recall-selection-stats --projects="${PROJECTS[*]}" --subsets="${SUBSETS[*]}" --datatypes="${DATATYPES[*]}"
        python -m pts.main make_tables --which=first-failure-auc-recall-selection-table --projects="${PROJECTS[*]}" --subsets="${SUBSETS[*]}" --datatypes="${DATATYPES[*]}"
}

function make_selection_time() {
         python -m pts.main make_tables --which=selection-time-stats --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}" --rtstools="${RTSTOOLS[*]}"
        python -m pts.main make_tables --which=selection-time-table --projects="${PROJECTS[*]}" --datatypes="${DATATYPES[*]}" --rtstools="${RTSTOOLS[*]}"
}

function make_bm25_perfect() {
        python -m pts.main make_tables --which=bm25-perfect-numbers
        python -m pts.main make_tables --which=bm25-perfect-table --projects="${PROJECTS[*]}"
}

function make_execution_time() {
        python -m pts.main collect_metrics --which=select-from-subset-execution-time --projects="${PROJECTS[*]}" --models="${DATATYPES[*]}"
}

function make_execution_time_change_selection_time() {
        python -m pts.main collect_metrics --which=select-from-subset-execution-time-change-selection-time --projects="${PROJECTS[*]}" --models="${DATATYPES[*]}"
}

function make_numbers_EALRTS(){
        python -m pts.main make_tables --which=EALRTS-numbers --projects="${PROJECTS[*]}"
}

function raw_dataset_stats_table() {
        python -m pts.main make_tables --which=dataset-raw-stats
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
