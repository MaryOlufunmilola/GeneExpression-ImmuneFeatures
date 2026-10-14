# preprocess_loco.R
#
# Leave-one-cohort-out (LOCO) inputs for deepImmune, rebuilt from the
# Data11 compendium (expression/, follow_up/, Immune_features/ per dataset).
#
# Replaces the data-preparation part of Funmi_proj1.r. Differences:
#   * datasets are matched by file name, never by list position
#   * columns are addressed by name, never by number (no df[, 711])
#   * duplicate gene symbols are collapsed by highest mean, not alphabetically
#   * ICB cohorts: pre-treatment samples only; response coded 1 = CR/PR,
#     0 = SD/PD, NA = NE/unknown; binary traits coded 0/1 (not 1/2)
#   * no imputation across cohorts and no zero-filling of unmeasured genes:
#     each fold uses only genes measured in every cohort of that fold
#   * normalization is within each cohort (z-score per gene), so the held-out
#     cohort never influences the training data
#   * one folder per held-out cohort with dataset_train.txt (training
#     cohorts), dataset_val.txt (seeded patient-level split of the training
#     cohorts, for tuning) and dataset_test.txt (held-out cohort only)
#
# After running:  python scripts/check_splits.py <out_dir>/folds
#
# Requires: data.table; biomaRt only if no cached annotation file exists.

suppressPackageStartupMessages(library(data.table))

## ---------------------------------------------------------------- settings
cfg <- list(
  data_dir        = "Data11",                 # contains */expression/*, */follow_up/*
  out_dir         = "Data11/loco",
  annotation_rds  = "Data11/ensembl_annotation.rds",   # cached biomaRt table
  seed            = 230,
  val_fraction    = 0.15,                     # of training patients, for tuning
  min_dataset_size = 10,                      # labelled pre-treatment patients per dataset file
  min_test_size   = 20,                       # per held-out study
  # Panel-based cohorts measure few genes and shrink every fold's gene set.
  exclude_dataset_pattern = "Nanostring|Oncomine",
  # Auxiliary (non-ICB) data added to training files for multi-task targets.
  # TCGA only by default: PRECOG microarrays cover fewer genes.
  auxiliary_pattern = "^TCGA-",
  icb_exclude_pattern = "^TCGA-|^PRECOG-|^METABRIC",
  survival_cols   = c("OS", "OS_EVENT")
)
set.seed(cfg$seed)

## ---------------------------------------------------------------- helpers
dataset_key <- function(path) sub("\\.(txt|tsv|csv)(\\.gz)?$", "", basename(path))

# Study = text before the first underscore (e.g. "Riaz2017" for all Riaz files).
# Sub-cohorts of one study are always held out together.
study_of <- function(ds) sub("_.*$", "", ds)

clean_names <- function(x) {
  x <- toupper(x)
  x <- gsub("[()]", "", x)
  gsub("-|[.]| ", "_", x)
}

first_present <- function(candidates, cols) {
  hit <- intersect(candidates, cols)
  if (length(hit)) hit[1] else NA_character_
}

load_annotation <- function(path) {
  if (file.exists(path)) return(readRDS(path))
  if (!requireNamespace("biomaRt", quietly = TRUE))
    stop("No cached annotation at ", path, " and biomaRt is not installed")
  mart <- biomaRt::useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")
  ann <- biomaRt::getBM(attributes = c("entrezgene_id", "ensembl_gene_id", "external_gene_name"),
                        mart = mart)
  saveRDS(ann, path)
  ann
}

# Expression file (genes in rows, first column = gene ID) -> samples x genes matrix
read_expression <- function(path, ann) {
  dt <- fread(path, data.table = FALSE)
  id_col <- colnames(dt)[1]
  ids <- as.character(dt[[1]])
  if (id_col == "EntrezID") {
    symbols <- ann$external_gene_name[match(ids, as.character(ann$entrezgene_id))]
  } else if (id_col == "Ensembl_id") {
    symbols <- ann$external_gene_name[match(sub("\\..*$", "", ids), ann$ensembl_gene_id)]
  } else {
    symbols <- ids
  }
  mat <- as.matrix(dt[, -1, drop = FALSE])
  storage.mode(mat) <- "double"
  keep <- !is.na(symbols) & symbols != ""
  mat <- mat[keep, , drop = FALSE]; symbols <- symbols[keep]
  # duplicate symbols: keep the most highly expressed row
  ord <- order(symbols, -rowMeans(mat, na.rm = TRUE))
  mat <- mat[ord, , drop = FALSE]; symbols <- symbols[ord]
  dup <- duplicated(symbols)
  mat <- mat[!dup, , drop = FALSE]
  rownames(mat) <- symbols[!dup]
  t(mat)
}

# log2-transform if the data look unlogged, then z-score each gene within the cohort
normalize_within_cohort <- function(x) {
  if (max(x, na.rm = TRUE) > 100) x <- log2(pmax(x, 0) + 1)
  mu <- colMeans(x, na.rm = TRUE)
  sdv <- apply(x, 2, sd, na.rm = TRUE)
  sdv[!is.finite(sdv) | sdv == 0] <- 1
  z <- sweep(sweep(x, 2, mu, "-"), 2, sdv, "/")
  z[is.na(z)] <- 0          # missing value within a measured gene -> cohort mean
  z
}

code_response <- function(x) {
  x <- toupper(trimws(as.character(x)))
  out <- rep(NA_real_, length(x))
  # Already-harmonized files code response as 1 (responder, CR/PR) / 0 (SD/PD)
  out[x %in% c("1", "1.0", "TRUE")] <- 1
  out[x %in% c("0", "0.0", "FALSE")] <- 0
  out[x %in% c("CR", "PR", "COMPLETE RESPONSE", "PARTIAL RESPONSE", "R", "RESPONDER")] <- 1
  out[x %in% c("SD", "PD", "STABLE DISEASE", "PROGRESSIVE DISEASE", "NR", "NON-RESPONDER", "NONRESPONDER")] <- 0
  out
}

# Sex: keep an existing 0/1 coding as it is; map text labels to 1 = male, 0 = female.
code_sex <- function(x) {
  v <- toupper(trimws(as.character(x)))
  v[v %in% c("", "NA", "NAN")] <- NA
  if (all(v[!is.na(v)] %in% c("0", "1", "0.0", "1.0"))) return(as.numeric(v))
  out <- rep(NA_real_, length(v))
  out[v %in% c("MALE", "M")] <- 1
  out[v %in% c("FEMALE", "F")] <- 0
  out
}

code_binary <- function(x, positive) {
  x <- toupper(trimws(as.character(x)))
  ifelse(is.na(x) | x == "" | x == "NA", NA_real_, as.numeric(x %in% toupper(positive)))
}

# column whose values are Pre/Post/On (biopsy timing), if any
timing_column <- function(clin) {
  for (cc in colnames(clin)) {
    v <- unique(toupper(trimws(as.character(clin[[cc]]))))
    v <- v[!is.na(v) & v != ""]
    if (length(v) && all(v %in% c("PRE", "POST", "ON", "EDT", "ON-TREATMENT"))) return(cc)
  }
  NA_character_
}

read_clinical <- function(path, is_tcga) {
  clin <- fread(path, data.table = FALSE)
  colnames(clin) <- clean_names(colnames(clin))
  sample_col  <- first_present(c("SAMPLE", "ARRAY", "PATIENT", "ID"), colnames(clin))
  if (is.na(sample_col)) sample_col <- colnames(clin)[1]
  patient_col <- first_present(c("PATIENT", "PATIENT_ID", "SUBJECT"), colnames(clin))
  clin$SAMPLE_ID  <- gsub("-", "_", as.character(clin[[sample_col]]))
  clin$PATIENT_ID <- if (is.na(patient_col)) clin$SAMPLE_ID else gsub("-", "_", as.character(clin[[patient_col]]))
  if (is_tcga) clin$PATIENT_ID <- substr(clin$PATIENT_ID, 1, 12)
  clin
}

## ---------------------------------------------------------------- load
files <- list.files(cfg$data_dir, recursive = TRUE, full.names = TRUE)
expr_files <- grep("/expression/", files, value = TRUE)
clin_files <- grep("/follow_up/", files, value = TRUE)
names(expr_files) <- dataset_key(expr_files)
names(clin_files) <- dataset_key(clin_files)

missing_clin <- setdiff(names(expr_files), names(clin_files))
if (length(missing_clin))
  warning("No follow_up file for: ", paste(missing_clin, collapse = ", "))
datasets <- intersect(names(expr_files), names(clin_files))
datasets <- datasets[!grepl(cfg$exclude_dataset_pattern, datasets)]
datasets <- datasets[!grepl("_Post$|_post$", datasets)]   # post-treatment-only files

ann <- load_annotation(cfg$annotation_rds)
icb_sets <- datasets[!grepl(cfg$icb_exclude_pattern, datasets)]
aux_sets <- datasets[grepl(cfg$auxiliary_pattern, datasets)]

cohorts <- list()
for (ds in c(icb_sets, aux_sets)) {
  is_icb <- ds %in% icb_sets
  expr <- read_expression(expr_files[[ds]], ann)
  rownames(expr) <- gsub("-", "_", rownames(expr))
  clin <- read_clinical(clin_files[[ds]], is_tcga = grepl("^TCGA-", ds))
  clin <- clin[clin$SAMPLE_ID %in% rownames(expr), , drop = FALSE]
  clin <- clin[!duplicated(clin$SAMPLE_ID), , drop = FALSE]

  if (is_icb) {
    tcol <- timing_column(clin)
    if (!is.na(tcol)) clin <- clin[toupper(trimws(clin[[tcol]])) %in% "PRE", , drop = FALSE]
    resp_col <- first_present(c("BOR", "BEST_OVERALL_RESPONSE", "RECIST", "RESPONSE", "RESPONDER"),
                              colnames(clin))
    if (is.na(resp_col)) { warning(ds, ": no response column, skipped"); next }
    clin$RESPONSE <- code_response(clin[[resp_col]])
    clin <- clin[!is.na(clin$RESPONSE), , drop = FALSE]
    # one sample per patient (pre-treatment); keeps the first if several
    clin <- clin[!duplicated(clin$PATIENT_ID), , drop = FALSE]
    if (nrow(clin) < cfg$min_dataset_size) {
      message(ds, ": ", nrow(clin), " labelled pre-treatment patients (< ", cfg$min_dataset_size, "), skipped")
      next
    }
  } else {
    clin$RESPONSE <- NA_real_
  }

  # 0/1 binary traits by column name (only if present)
  if ("SEX" %in% colnames(clin)) clin$SEX_MALE <- code_sex(clin$SEX)
  if ("GENDER" %in% colnames(clin)) clin$SEX_MALE <- code_sex(clin$GENDER)

  # Survival time units differ between studies (days, months, years). That is fine:
  # deepImmune rank-normalizes survival time within cohort, and the survival script
  # stratifies by cohort. Never pool raw times across cohorts.
  for (sc in cfg$survival_cols) {
    clin[[sc]] <- if (sc %in% colnames(clin)) suppressWarnings(as.numeric(clin[[sc]])) else NA_real_
  }

  cohorts[[ds]] <- list(
    name = ds, is_icb = is_icb,
    expr = normalize_within_cohort(expr[clin$SAMPLE_ID, , drop = FALSE]),
    clin = clin
  )
}
icb_sets <- intersect(icb_sets, names(cohorts))
aux_sets <- intersect(aux_sets, names(cohorts))
message(length(icb_sets), " ICB cohorts and ", length(aux_sets), " auxiliary datasets loaded")

## ------------------------------------------- immune-feature targets (TCGA)
# Optional: TCGA immune features as continuous multi-task targets, merged by
# patient ID. Uses the same supplementary tables as Funmi_proj1.r.
immune_targets <- NULL
supp <- file.path(cfg$data_dir, sprintf("DatafileS%d.xlsx", 1:4))
if (all(file.exists(supp)) && requireNamespace("readxl", quietly = TRUE)) {
  rd <- function(f, sheet) as.data.frame(readxl::read_excel(f, sheet = sheet))
  tabs <- list(rd(supp[1], "EMT_Immune features"), rd(supp[2], "TCGA features"),
               rd(supp[3], "TCGA features"), rd(supp[4], "TCGA features"), rd(supp[4], "TCGA_Freeman2022"))
  tabs <- lapply(seq_along(tabs), function(i) {
    x <- tabs[[i]]
    x$Samples <- gsub("-", "_", substr(x$Samples, 1, 12))
    x <- x[!duplicated(x$Samples), , drop = FALSE]
    keep <- vapply(x, is.numeric, logical(1)); keep["Samples"] <- TRUE
    x <- x[, keep, drop = FALSE]
    colnames(x)[colnames(x) != "Samples"] <- paste0("IMM_", clean_names(colnames(x)[colnames(x) != "Samples"]), "_S", i)
    x
  })
  immune_targets <- Reduce(function(a, b) merge(a, b, by = "Samples", all = TRUE), tabs)
  colnames(immune_targets)[1] <- "PATIENT_ID"
  message(ncol(immune_targets) - 1, " immune-feature targets loaded")
}

## ---------------------------------------------------------------- folds
assemble <- function(sets, genes, extra_cols) {
  parts <- lapply(sets, function(ds) {
    co <- cohorts[[ds]]
    x <- as.data.frame(co$expr[, genes, drop = FALSE])
    lab <- co$clin[, c("PATIENT_ID", "RESPONSE", cfg$survival_cols,
                       intersect("SEX_MALE", colnames(co$clin))), drop = FALSE]
    out <- data.frame(cancertype = rep(ds, nrow(x)), x, lab, check.names = FALSE, stringsAsFactors = FALSE)
    out$COHORT <- study_of(ds)
    out$PATIENT_ID <- paste(ds, out$PATIENT_ID, sep = ":")   # unique across cohorts
    out
  })
  out <- rbindlist(parts, fill = TRUE)
  if (!is.null(immune_targets)) {
    raw_id <- sub("^[^:]*:", "", out$PATIENT_ID)
    imm <- immune_targets[match(raw_id, immune_targets$PATIENT_ID), -1, drop = FALSE]
    out <- cbind(out, imm)
  }
  for (cc in setdiff(extra_cols, colnames(out))) set(out, j = cc, value = NA_real_)
  label_cols <- setdiff(colnames(out), c("cancertype", genes, "PATIENT_ID", "COHORT"))
  setcolorder(out, c("cancertype", genes, label_cols, "PATIENT_ID", "COHORT"))
  out
}

dir.create(file.path(cfg$out_dir, "folds"), recursive = TRUE, showWarnings = FALSE)
summary_rows <- list()
for (held_out in unique(study_of(icb_sets))) {
  test_sets  <- icb_sets[study_of(icb_sets) == held_out]
  train_sets <- c(setdiff(icb_sets, test_sets), aux_sets)
  n_test <- sum(vapply(test_sets, function(ds) nrow(cohorts[[ds]]$clin), integer(1)))
  if (n_test < cfg$min_test_size) {
    message(held_out, ": ", n_test, " test patients (< ", cfg$min_test_size, "), not used as a test fold")
    next
  }
  genes <- Reduce(intersect, lapply(c(train_sets, test_sets), function(ds) colnames(cohorts[[ds]]$expr)))
  genes <- sort(genes)

  train <- assemble(train_sets, genes, character(0))
  test  <- assemble(test_sets, genes, colnames(train))
  test  <- test[, colnames(train), with = FALSE]   # identical columns, same order

  # seeded patient-level validation split inside the training cohorts
  pts <- unique(train$PATIENT_ID)
  val_pts <- sample(pts, round(cfg$val_fraction * length(pts)))
  val <- train[PATIENT_ID %in% val_pts]
  train <- train[!PATIENT_ID %in% val_pts]

  fold_dir <- file.path(cfg$out_dir, "folds", held_out)
  dir.create(fold_dir, showWarnings = FALSE)
  fwrite(train, file.path(fold_dir, "dataset_train.txt"), sep = "\t", na = "NA")
  fwrite(val,   file.path(fold_dir, "dataset_val.txt"),   sep = "\t", na = "NA")
  fwrite(test,  file.path(fold_dir, "dataset_test.txt"),  sep = "\t", na = "NA")

  # deepImmune drops the first (cancertype) column and indexes from 0
  idx <- data.table(column = colnames(train)[-1], index = seq_len(ncol(train) - 1) - 1)
  fwrite(idx, file.path(fold_dir, "column_index.tsv"), sep = "\t")

  summary_rows[[held_out]] <- data.table(
    held_out_cohort = held_out, n_genes = length(genes),
    n_test = nrow(test), test_responders = sum(test$RESPONSE == 1, na.rm = TRUE),
    n_train = nrow(train), n_val = nrow(val)
  )
}

cohort_table <- rbindlist(lapply(icb_sets, function(ds) {
  cl <- cohorts[[ds]]$clin
  data.table(study = study_of(ds), dataset = ds, n_pretreatment = nrow(cl), responders = sum(cl$RESPONSE == 1),
             response_rate = round(mean(cl$RESPONSE == 1), 3),
             has_OS = all(c("OS", "OS_EVENT") %in% colnames(cl)) && any(!is.na(cl$OS)))
}))
fwrite(cohort_table, file.path(cfg$out_dir, "table1_cohorts.tsv"), sep = "\t")
fwrite(rbindlist(summary_rows), file.path(cfg$out_dir, "fold_summary.tsv"), sep = "\t")
writeLines(capture.output(sessionInfo()), file.path(cfg$out_dir, "sessionInfo.txt"))
message("Wrote ", length(summary_rows), " folds to ", file.path(cfg$out_dir, "folds"))
