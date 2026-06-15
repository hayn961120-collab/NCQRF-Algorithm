library(grf)
library(jsonlite)
library(openxlsx)

# 설정
setwd("C:/quantile")

shared_root <- "./realdata_shared_randomsplit_repeat100"
out_root <- "./grf/randomsplit_100"
tau <- c(0.1, 0.3, 0.5, 0.7, 0.9)
dataset_name <- "airfoil"

STANDARDIZE_Y <- FALSE

dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

# 유틸
pinball_loss <- function(y_true, y_pred, tau) {
  u <- y_true - y_pred
  mean(ifelse(u >= 0, tau * u, (tau - 1) * u))
}

composite_pinball <- function(y_true, pred_mat, tau) {
  per_tau <- sapply(seq_along(tau), function(j) {
    pinball_loss(y_true, pred_mat[, j], tau[j])
  })
  list(comp = mean(per_tau), per_tau = as.numeric(per_tau))
}

crossing_percentage <- function(pred_mat) {
  if (is.null(dim(pred_mat)) || ncol(pred_mat) < 2) return(0)
  prev <- pred_mat[, -ncol(pred_mat), drop = FALSE]
  nxt  <- pred_mat[, -1, drop = FALSE]
  mean(prev > nxt) * 100
}

mean_std_se <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  m <- mean(x)
  s <- if (n > 1) sd(x) else 0
  se <- if (n > 0) s / sqrt(n) else 0
  list(mean = m, std = s, se = se, n = n)
}

make_mtry_grid <- function(p_dim) {
  if (p_dim == 1) return(1)
  
  out <- unique(c(
    floor(sqrt(p_dim)),
    floor(log2(p_dim)),
    floor(0.25 * p_dim),
    p_dim
  ))
  
  out <- out[out >= 1]
  return(out)
}

make_min_samples_leaf_grid <- function(train_n) {
  c(1, 2, 5, 10, 15, 20, 25, 30)
}

save_summary_excel <- function(summary, out_xlsx, model_name = "grf") {
  wb <- createWorkbook()
  
  df_tau <- data.frame(
    tau = summary$tau,
    mean_pinball = summary$mean_tau,
    SE_pinball = summary$se_tau
  )
  names(df_tau)[2] <- paste0(model_name, "_mean_pinball")
  names(df_tau)[3] <- paste0(model_name, "_SE_pinball")
  
  df_comp <- data.frame(
    n_eval = summary$n_eval,
    comp_mean = summary$comp_mean,
    comp_SE = summary$comp_se,
    cross_mean = summary$cross_mean,
    cross_SE = summary$cross_se
  )
  names(df_comp)[2] <- paste0(model_name, "_comp_mean")
  names(df_comp)[3] <- paste0(model_name, "_comp_SE")
  names(df_comp)[4] <- paste0(model_name, "_cross_mean")
  names(df_comp)[5] <- paste0(model_name, "_cross_SE")
  
  addWorksheet(wb, "per_tau")
  writeData(wb, "per_tau", df_tau)
  
  addWorksheet(wb, "composite")
  writeData(wb, "composite", df_comp)
  
  saveWorkbook(wb, out_xlsx, overwrite = TRUE)
}

plot_tau_curve <- function(summary, out_png, tag, model_name = "grf") {
  png(out_png, width = 900, height = 700, res = 150)
  
  y_min <- min(summary$mean_tau - summary$se_tau)
  y_max <- max(summary$mean_tau + summary$se_tau)
  
  plot(
    x = summary$tau,
    y = summary$mean_tau,
    type = "b",
    pch = 19,
    xlab = "tau",
    ylab = "Mean pinball loss (TEST)",
    main = paste0(tag, ": ", model_name, " per-tau mean pinball loss (±1 SE)"),
    ylim = c(y_min, y_max)
  )
  
  arrows(
    x0 = summary$tau,
    y0 = summary$mean_tau - summary$se_tau,
    x1 = summary$tau,
    y1 = summary$mean_tau + summary$se_tau,
    angle = 90,
    code = 3,
    length = 0.05
  )
  
  dev.off()
}

# Python 저장 random split 데이터 로드
load_saved_dataset_random <- function(dataset_name,
                                      shared_root = "./realdata_shared_randomsplit") {
  dataset_dir <- file.path(shared_root, dataset_name)
  
  data_csv_path <- file.path(dataset_dir, paste0(dataset_name, "_full_data.csv"))
  split_json_path <- file.path(dataset_dir, paste0(dataset_name, "_splits_random.json"))
  meta_json_path <- file.path(dataset_dir, paste0(dataset_name, "_meta_random.json"))
  
  if (!file.exists(data_csv_path)) {
    stop("full_data.csv not found: ", data_csv_path)
  }
  if (!file.exists(split_json_path)) {
    stop("splits_random.json not found: ", split_json_path)
  }
  
  df <- read.csv(data_csv_path, check.names = FALSE)
  split_info <- fromJSON(split_json_path, simplifyVector = FALSE)
  
  meta <- NULL
  if (file.exists(meta_json_path)) {
    meta <- fromJSON(meta_json_path, simplifyVector = FALSE)
  }
  
  feature_cols <- setdiff(colnames(df), c("row_id", "y"))
  
  list(
    df = df,
    feature_cols = feature_cols,
    split_info = split_info,
    meta = meta
  )
}

# GRF tuning
fit_and_eval_grf <- function(Xtr, ytr, Xva, yva, tau,
                             num.trees,
                             mtry,
                             min.node.size,
                             sample.fraction,
                             honesty,
                             seed) {
  fit <- quantile_forest(
    X = Xtr,
    Y = ytr,
    quantiles = tau,
    num.trees = num.trees,
    regression.splitting = FALSE,
    mtry = mtry,
    min.node.size = min.node.size,
    sample.fraction = sample.fraction,
    honesty = honesty,
    seed = seed
  )
  
  pred_valid <- predict(fit, newdata = Xva, quantiles = tau)$predictions
  val_res <- composite_pinball(yva, pred_valid, tau)
  
  list(
    valid_comp = val_res$comp,
    valid_per_tau = val_res$per_tau
  )
}

tune_grf_once <- function(X_train, y_train,
                          X_valid, y_valid,
                          tau,
                          repeat_id,
                          seed = 100) {
  p_dim <- ncol(X_train)
  train_n <- nrow(X_train)
  
  mtry_grid <- make_mtry_grid(p_dim)
  min_node_grid <- make_min_samples_leaf_grid(train_n)
  
  grid <- expand.grid(
    num.trees = c(1000),
    min.node.size = min_node_grid,
    sample.fraction = c(0.632),
    honesty = c(FALSE),
    mtry = mtry_grid
  )
  
  tuning_rows <- vector("list", nrow(grid))
  best_val <- Inf
  best_idx <- NA
  
  for (i in seq_len(nrow(grid))) {
    params <- grid[i, ]
    
    res <- fit_and_eval_grf(
      Xtr = X_train,
      ytr = y_train,
      Xva = X_valid,
      yva = y_valid,
      tau = tau,
      num.trees = params$num.trees,
      mtry = params$mtry,
      min.node.size = params$min.node.size,
      sample.fraction = params$sample.fraction,
      honesty = params$honesty,
      seed = seed + repeat_id * 1000 + i
    )
    
    tuning_rows[[i]] <- data.frame(
      repeat_id = repeat_id,
      candidate_id = i,
      num.trees = params$num.trees,
      mtry = params$mtry,
      min.node.size = params$min.node.size,
      sample.fraction = params$sample.fraction,
      honesty = params$honesty,
      valid_comp = res$valid_comp
    )
    
    if (res$valid_comp < best_val) {
      best_val <- res$valid_comp
      best_idx <- i
    }
  }
  
  list(
    best_params = as.list(grid[best_idx, ]),
    best_valid_comp = best_val,
    tuning_df = do.call(rbind, tuning_rows)
  )
}

# repeat 1개 실행
run_one_eval_grf <- function(df,
                             feature_cols,
                             split_one,
                             tau,
                             standardize_y = TRUE,
                             seed = 100) {
  repeat_id <- split_one[["repeat"]]
  
  inner_train_ids <- unlist(split_one[["inner_train_row_id"]])
  inner_valid_ids <- unlist(split_one[["inner_valid_row_id"]])
  train_ids <- unlist(split_one[["train_row_id"]])
  test_ids <- unlist(split_one[["test_row_id"]])
  
  df_inner_train <- df[df$row_id %in% inner_train_ids, , drop = FALSE]
  df_inner_valid <- df[df$row_id %in% inner_valid_ids, , drop = FALSE]
  df_train <- df[df$row_id %in% train_ids, , drop = FALSE]
  df_test <- df[df$row_id %in% test_ids, , drop = FALSE]
  
  X_inner_train <- as.matrix(df_inner_train[, feature_cols, drop = FALSE])
  X_inner_valid <- as.matrix(df_inner_valid[, feature_cols, drop = FALSE])
  X_train <- as.matrix(df_train[, feature_cols, drop = FALSE])
  X_test <- as.matrix(df_test[, feature_cols, drop = FALSE])
  
  y_inner_train_raw <- df_inner_train$y
  y_inner_valid_raw <- df_inner_valid$y
  y_train_raw <- df_train$y
  y_test_raw <- df_test$y
  
  if (standardize_y) {
    y_mean <- mean(y_train_raw)
    y_std <- sd(y_train_raw)
    
    if (is.na(y_std) || y_std == 0) {
      y_std <- 1
    }
    
    y_inner_train <- (y_inner_train_raw - y_mean) / y_std
    y_inner_valid <- (y_inner_valid_raw - y_mean) / y_std
    y_train <- (y_train_raw - y_mean) / y_std
    y_test <- (y_test_raw - y_mean) / y_std
  } else {
    y_mean <- NA_real_
    y_std <- NA_real_
    
    y_inner_train <- y_inner_train_raw
    y_inner_valid <- y_inner_valid_raw
    y_train <- y_train_raw
    y_test <- y_test_raw
  }
  
  tune_res <- tune_grf_once(
    X_train = X_inner_train,
    y_train = y_inner_train,
    X_valid = X_inner_valid,
    y_valid = y_inner_valid,
    tau = tau,
    repeat_id = repeat_id,
    seed = seed
  )
  
  best_params <- tune_res$best_params
  
  fit_final <- quantile_forest(
    X = X_train,
    Y = y_train,
    quantiles = tau,
    num.trees = best_params$num.trees,
    regression.splitting = FALSE,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    honesty = best_params$honesty,
    seed = seed + 10000 + repeat_id * 1000
  )
  
  pred_test <- predict(fit_final, newdata = X_test, quantiles = tau)$predictions
  
  test_res <- composite_pinball(y_test, pred_test, tau)
  test_cross <- crossing_percentage(pred_test)
  
  per_tau_df <- data.frame(
    repeat_id = repeat_id,
    tau = tau,
    test_pinball = test_res$per_tau
  )
  
  comp_df <- data.frame(
    repeat_id = repeat_id,
    n_train = nrow(df_train),
    n_test = nrow(df_test),
    n_inner_train = nrow(df_inner_train),
    n_inner_valid = nrow(df_inner_valid),
    test_comp = test_res$comp,
    test_cross = test_cross,
    best_valid_comp = tune_res$best_valid_comp,
    num.trees = best_params$num.trees,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    honesty = best_params$honesty,
    y_standardized = standardize_y,
    y_mean_train = y_mean,
    y_std_train = y_std
  )
  
  list(
    per_tau_df = per_tau_df,
    comp_df = comp_df,
    tuning_df = tune_res$tuning_df
  )
}

# summary
summarize_randomsplit_results <- function(per_tau_all,
                                          comp_all,
                                          tau,
                                          model_name = "grf") {
  mean_tau <- numeric(length(tau))
  std_tau <- numeric(length(tau))
  se_tau <- numeric(length(tau))
  
  for (j in seq_along(tau)) {
    vals <- per_tau_all$test_pinball[per_tau_all$tau == tau[j]]
    tmp <- mean_std_se(vals)
    mean_tau[j] <- tmp$mean
    std_tau[j] <- tmp$std
    se_tau[j] <- tmp$se
  }
  
  comp_stat <- mean_std_se(comp_all$test_comp)
  cross_stat <- mean_std_se(comp_all$test_cross)
  
  list(
    model_name = model_name,
    tau = tau,
    n_eval = nrow(comp_all),
    mean_tau = mean_tau,
    std_tau = std_tau,
    se_tau = se_tau,
    comp_mean = comp_stat$mean,
    comp_std = comp_stat$std,
    comp_se = comp_stat$se,
    cross_mean = cross_stat$mean,
    cross_std = cross_stat$std,
    cross_se = cross_stat$se
  )
}

# 메인 실행 함수
run_real_grf_from_saved_splits_random <- function(dataset_name,
                                                  tau,
                                                  shared_root = "./realdata_shared_randomsplit",
                                                  out_root = "./real_results_r_grf_randomsplit",
                                                  model_name = "grf",
                                                  standardize_y = TRUE,
                                                  seed = 100) {
  loaded <- load_saved_dataset_random(
    dataset_name = dataset_name,
    shared_root = shared_root
  )
  
  df <- loaded$df
  feature_cols <- loaded$feature_cols
  split_info <- loaded$split_info
  
  tag <- paste0(model_name, "_", dataset_name)
  dataset_out <- file.path(out_root, dataset_name)
  
  dir.create(dataset_out, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(dataset_out, "raw"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(dataset_out, "excel"), showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(dataset_out, "plots"), showWarnings = FALSE, recursive = TRUE)
  
  checkpoint_rds <- file.path(
    dataset_out,
    "raw",
    paste0(tag, "_checkpoint_randomsplit.rds")
  )
  
  if (file.exists(checkpoint_rds)) {
    ckpt <- readRDS(checkpoint_rds)
    per_tau_list <- ckpt$per_tau_list
    comp_list <- ckpt$comp_list
    tuning_list <- ckpt$tuning_list
    done_repeats <- ckpt$done_repeats
  } else {
    per_tau_list <- list()
    comp_list <- list()
    tuning_list <- list()
    done_repeats <- integer(0)
  }
  
  total_jobs <- length(split_info$splits)
  
  for (i in seq_along(split_info$splits)) {
    split_one <- split_info$splits[[i]]
    repeat_id <- split_one[["repeat"]]
    
    if (repeat_id %in% done_repeats) {
      cat("[SKIP] dataset =", dataset_name, "| repeat =", repeat_id, "\n")
      next
    }
    
    cat("\n====================================================\n")
    cat("dataset =", dataset_name,
        "| repeat =", repeat_id,
        "| job =", i, "/", total_jobs, "\n")
    cat("====================================================\n")
    
    one <- run_one_eval_grf(
      df = df,
      feature_cols = feature_cols,
      split_one = split_one,
      tau = tau,
      standardize_y = standardize_y,
      seed = seed
    )
    
    key <- as.character(repeat_id)
    per_tau_list[[key]] <- one$per_tau_df
    comp_list[[key]] <- one$comp_df
    tuning_list[[key]] <- one$tuning_df
    
    done_repeats <- sort(unique(c(done_repeats, repeat_id)))
    
    saveRDS(
      list(
        per_tau_list = per_tau_list,
        comp_list = comp_list,
        tuning_list = tuning_list,
        done_repeats = done_repeats
      ),
      checkpoint_rds
    )
    
    cat("repeat =", repeat_id,
        "| test comp =", sprintf("%.6f", one$comp_df$test_comp),
        "| cross =", sprintf("%.6f", one$comp_df$test_cross), "\n")
    cat("[CHECKPOINT SAVED] ", checkpoint_rds, "\n", sep = "")
  }
  
  per_tau_all <- do.call(rbind, per_tau_list)
  comp_all <- do.call(rbind, comp_list)
  tuning_all <- do.call(rbind, tuning_list)
  
  per_tau_all <- per_tau_all[order(per_tau_all$repeat_id, per_tau_all$tau), ]
  comp_all <- comp_all[order(comp_all$repeat_id), ]
  tuning_all <- tuning_all[order(tuning_all$repeat_id, tuning_all$candidate_id), ]
  
  summary <- summarize_randomsplit_results(
    per_tau_all = per_tau_all,
    comp_all = comp_all,
    tau = tau,
    model_name = model_name
  )
  
  raw_csv_per_tau <- file.path(dataset_out, "raw", paste0(tag, "_per_tau_raw.csv"))
  raw_csv_comp <- file.path(dataset_out, "raw", paste0(tag, "_comp_raw.csv"))
  raw_csv_tuning <- file.path(dataset_out, "raw", paste0(tag, "_tuning_raw.csv"))
  out_xlsx <- file.path(dataset_out, "excel", paste0(tag, ".xlsx"))
  out_png <- file.path(dataset_out, "plots", paste0(tag, "_tau_curve.png"))
  out_rds <- file.path(dataset_out, "raw", paste0(tag, "_raw_result.rds"))
  
  write.csv(per_tau_all, raw_csv_per_tau, row.names = FALSE)
  write.csv(comp_all, raw_csv_comp, row.names = FALSE)
  write.csv(tuning_all, raw_csv_tuning, row.names = FALSE)
  
  save_summary_excel(summary, out_xlsx, model_name = model_name)
  plot_tau_curve(summary, out_png, tag = tag, model_name = model_name)
  
  saveRDS(
    list(
      summary = summary,
      per_tau_all = per_tau_all,
      comp_all = comp_all,
      tuning_all = tuning_all,
      standardize_y = standardize_y,
      shared_root = shared_root
    ),
    out_rds
  )
  
  cat("\n================ FINAL SUMMARY ================\n")
  cat("dataset =", dataset_name, "\n")
  cat("n_eval =", summary$n_eval, "\n")
  cat("Y standardized =", standardize_y, "\n")
  cat("comp_mean =", summary$comp_mean, "\n")
  cat("comp_SE =", summary$comp_se, "\n")
  cat("cross_mean =", summary$cross_mean, "\n")
  cat("cross_SE =", summary$cross_se, "\n")
  cat("[Shared root]     ", shared_root, "\n", sep = "")
  cat("[Saved per_tau]   ", raw_csv_per_tau, "\n", sep = "")
  cat("[Saved comp]      ", raw_csv_comp, "\n", sep = "")
  cat("[Saved tuning]    ", raw_csv_tuning, "\n", sep = "")
  cat("[Saved Excel]     ", out_xlsx, "\n", sep = "")
  cat("[Saved Plot]      ", out_png, "\n", sep = "")
  cat("[Saved RDS]       ", out_rds, "\n", sep = "")
  cat("===============================================\n")
}

# 실행
run_real_grf_from_saved_splits_random(
  dataset_name = dataset_name,
  tau = tau,
  shared_root = shared_root,
  out_root = out_root,
  model_name = "grf",
  standardize_y = STANDARDIZE_Y,
  seed = 100
)