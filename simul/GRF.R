library(grf)
library(openxlsx)

setwd("C:/quantile")

scenario_id <- 1
p <- 1
train_n <- 1000
mc_repeats <- 100
tau <- c(0.1, 0.3, 0.5, 0.7, 0.9)

input_dir <- "0414_grf_input_exact"
out_root  <- "grf/n1000/fal_result_grf"

dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "raw"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "excel"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "plots"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_root, "seed_cache"), showWarnings = FALSE, recursive = TRUE)

# 1. 함수들 ------------------------------------------------------------

pinball_loss <- function(y_true, y_pred, tau) {
  u <- y_true - y_pred
  mean(ifelse(u >= 0, tau * u, (tau - 1) * u))
}

composite_pinball <- function(y_true, pred_mat, tau) {
  per_tau <- sapply(seq_along(tau), function(j) {
    pinball_loss(y_true, pred_mat[, j], tau[j])
  })
  list(
    comp = mean(per_tau),
    per_tau = as.numeric(per_tau)
  )
}

crossing_percentage <- function(pred_mat) {
  if (is.null(dim(pred_mat)) || ncol(pred_mat) < 2) {
    return(0)
  }
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

load_split_data <- function(scenario_id, p, train_n, seed, input_dir) {
  base <- paste0("s", scenario_id, "_p", p, "_n", train_n, "_seed", seed)
  
  train_path <- file.path(input_dir, paste0(base, "_train.csv"))
  valid_path <- file.path(input_dir, paste0(base, "_valid.csv"))
  test_path  <- file.path(input_dir, paste0(base, "_test.csv"))
  
  if (!file.exists(train_path)) stop("파일 없음: ", train_path)
  if (!file.exists(valid_path)) stop("파일 없음: ", valid_path)
  if (!file.exists(test_path))  stop("파일 없음: ", test_path)
  
  train_df <- read.csv(train_path)
  valid_df <- read.csv(valid_path)
  test_df  <- read.csv(test_path)
  
  X_train <- as.matrix(train_df[, colnames(train_df) != "y", drop = FALSE])
  y_train <- train_df$y
  
  X_valid <- as.matrix(valid_df[, colnames(valid_df) != "y", drop = FALSE])
  y_valid <- valid_df$y
  
  X_test <- as.matrix(test_df[, colnames(test_df) != "y", drop = FALSE])
  y_test <- test_df$y
  
  list(
    X_train = X_train, y_train = y_train,
    X_valid = X_valid, y_valid = y_valid,
    X_test  = X_test,  y_test  = y_test
  )
}

make_mtry_grid <- function(p_dim) {
  if (p_dim == 1) {
    return(1)
  } else {
    out <- unique(c(
      floor(sqrt(p_dim)),
      floor(log2(p_dim)),
      floor(0.25 * p_dim),
      p_dim
    ))
    out <- out[out >= 1]
    return(out)
  }
}
make_min_node_size_grid <- function(train_n) {
  vals <- c(1, 2, 5, 10, 15, 20,25,30)
  return(vals)
}

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
    fit = fit,
    valid_comp = val_res$comp,
    valid_per_tau = val_res$per_tau
  )
}

tune_grf_once <- function(X_train, y_train, X_valid, y_valid, tau, seed, train_n) {
  p_dim <- ncol(X_train)
  mtry_grid <- make_mtry_grid(p_dim)
  min_node_size_grid <- make_min_node_size_grid(train_n)
  
  grid <- expand.grid(
    num.trees = c(1000),
    min.node.size = min_node_size_grid,
    sample.fraction = c(0.632),
    honesty = c(FALSE),
    mtry = mtry_grid
  )
  
  tuning_rows <- vector("list", nrow(grid))
  
  best_val <- Inf
  best_idx <- NA
  
  cat("----------------------------------------------------\n")
  cat("seed =", seed, "| candidates =", nrow(grid), "\n")
  cat("min.node.size grid =", paste(min_node_size_grid, collapse = ", "), "\n")
  
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
      seed = seed
    )
    
    tuning_rows[[i]] <- data.frame(
      seed = seed,
      candidate_id = i,
      num.trees = params$num.trees,
      mtry = params$mtry,
      min.node.size = params$min.node.size,
      sample.fraction = params$sample.fraction,
      honesty = params$honesty,
      valid_comp = res$valid_comp
    )
    
    cat("[", i, "/", nrow(grid), "] ",
        "mtry=", params$mtry,
        ", min.node.size=", params$min.node.size,
        ", sample.fraction=", params$sample.fraction,
        " -> valid=", sprintf("%.6f", res$valid_comp), "\n", sep = "")
    
    if (res$valid_comp < best_val) {
      best_val <- res$valid_comp
      best_idx <- i
    }
  }
  
  best_params <- as.list(grid[best_idx, ])
  tuning_df <- do.call(rbind, tuning_rows)
  
  list(
    best_params = best_params,
    best_valid_comp = best_val,
    tuning_df = tuning_df
  )
}

run_one_seed_grf <- function(scenario_id, p, train_n, seed, tau, input_dir) {
  dat <- load_split_data(
    scenario_id = scenario_id,
    p = p,
    train_n = train_n,
    seed = seed,
    input_dir = input_dir
  )
  
  X_train <- dat$X_train
  y_train <- dat$y_train
  X_valid <- dat$X_valid
  y_valid <- dat$y_valid
  X_test  <- dat$X_test
  y_test  <- dat$y_test
  
  tune_res <- tune_grf_once(
    X_train = X_train,
    y_train = y_train,
    X_valid = X_valid,
    y_valid = y_valid,
    tau = tau,
    seed = seed,
    train_n = train_n
  )
  
  best_params <- tune_res$best_params
  
  fit_final <- quantile_forest(
    X = X_train,
    Y = y_train,
    quantiles = tau,
    num.trees = best_params$num.trees,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    honesty = best_params$honesty,
    seed = seed
  )
  
  pred_test <- predict(fit_final, newdata = X_test, quantiles = tau)$predictions
  test_res <- composite_pinball(y_test, pred_test, tau)
  test_cross <- crossing_percentage(pred_test)
  
  per_tau_df <- data.frame(
    seed = seed,
    tau = tau,
    test_pinball = test_res$per_tau
  )
  
  comp_df <- data.frame(
    seed = seed,
    test_comp = test_res$comp,
    test_cross = test_cross,
    best_valid_comp = tune_res$best_valid_comp,
    num.trees = best_params$num.trees,
    mtry = best_params$mtry,
    min.node.size = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    honesty = best_params$honesty
  )
  
  list(
    per_tau_df = per_tau_df,
    comp_df = comp_df,
    tuning_df = tune_res$tuning_df
  )
}

summarize_mc_results <- function(per_tau_all, comp_all, tau, model_name = "grf") {
  mean_tau <- numeric(length(tau))
  std_tau  <- numeric(length(tau))
  se_tau   <- numeric(length(tau))
  
  for (j in seq_along(tau)) {
    vals <- per_tau_all$test_pinball[per_tau_all$tau == tau[j]]
    tmp <- mean_std_se(vals)
    mean_tau[j] <- tmp$mean
    std_tau[j]  <- tmp$std
    se_tau[j]   <- tmp$se
  }
  
  comp_stat  <- mean_std_se(comp_all$test_comp)
  cross_stat <- mean_std_se(comp_all$test_cross)
  
  summary <- list(
    model_name = model_name,
    tau = tau,
    n_mc = length(unique(comp_all$seed)),
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
  
  return(summary)
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
    n_mc = summary$n_mc,
    comp_mean = summary$comp_mean,
    comp_SE = summary$comp_se,
    cross_mean = summary$cross_mean,
    cross_SE = summary$cross_se
  )
  names(df_comp)[2] <- paste0(model_name, "_comp_mean")
  names(df_comp)[3] <- paste0(model_name, "_comp_SE")
  names(df_comp)[4] <- paste0(model_name, "_cross_mean")
  names(df_comp)[5] <- paste0(model_name, "_cross_SE")
  
  df_notes <- data.frame(
    note = c(
      "SE = sd(ddof=1) / sqrt(n_mc).",
      "per_tau: TEST set pinball loss summary.",
      "composite: TEST composite pinball summary.",
      "cross_mean: TEST crossing percentage summary."
    )
  )
  
  addWorksheet(wb, "per_tau")
  writeData(wb, "per_tau", df_tau)
  
  addWorksheet(wb, "composite")
  writeData(wb, "composite", df_comp)
  
  addWorksheet(wb, "notes")
  writeData(wb, "notes", df_notes)
  
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

save_raw_results_rds <- function(per_tau_all, comp_all, tuning_all, tau, out_rds, model_name = "grf") {
  per_tau_wide <- reshape(
    per_tau_all[, c("seed", "tau", "test_pinball")],
    idvar = "seed",
    timevar = "tau",
    direction = "wide"
  )
  per_tau_wide <- per_tau_wide[order(per_tau_wide$seed), ]
  
  res <- list(
    tau = tau,
    model = list(
      comp = comp_all$test_comp,
      per_tau = as.matrix(per_tau_wide[, -1, drop = FALSE])
    ),
    model_name = model_name,
    per_tau_raw = per_tau_all,
    comp_raw = comp_all,
    tuning_raw = tuning_all
  )
  saveRDS(res, out_rds)
}

save_one_seed_cache <- function(one, seed_cache_path) {
  saveRDS(one, seed_cache_path)
}

load_completed_seed_results <- function(seed_cache_dir, tag, mc_repeats) {
  per_tau_list <- list()
  comp_list <- list()
  tuning_list <- list()
  completed_seeds <- integer(0)
  
  for (seed in 0:(mc_repeats - 1)) {
    seed_cache_path <- file.path(seed_cache_dir, paste0(tag, "_seed", seed, ".rds"))
    if (file.exists(seed_cache_path)) {
      one <- readRDS(seed_cache_path)
      per_tau_list[[length(per_tau_list) + 1]] <- one$per_tau_df
      comp_list[[length(comp_list) + 1]] <- one$comp_df
      tuning_list[[length(tuning_list) + 1]] <- one$tuning_df
      completed_seeds <- c(completed_seeds, seed)
    }
  }
  
  list(
    completed_seeds = completed_seeds,
    per_tau_all = if (length(per_tau_list) > 0) do.call(rbind, per_tau_list) else NULL,
    comp_all    = if (length(comp_list) > 0) do.call(rbind, comp_list) else NULL,
    tuning_all  = if (length(tuning_list) > 0) do.call(rbind, tuning_list) else NULL
  )
}

# 2. 메인 실행 ---------------------------------------------------------

run_mc_grf <- function(scenario_id, p, train_n, mc_repeats, tau,
                       input_dir = "grf_input_exact",
                       out_root = "grf_mc_results",
                       model_name = "grf") {
  
  tag <- paste0(model_name, "_s", scenario_id, "_p", p)
  seed_cache_dir <- file.path(out_root, "seed_cache")
  dir.create(seed_cache_dir, showWarnings = FALSE, recursive = TRUE)
  
  loaded <- load_completed_seed_results(
    seed_cache_dir = seed_cache_dir,
    tag = tag,
    mc_repeats = mc_repeats
  )
  
  completed_seeds <- loaded$completed_seeds
  remaining_seeds <- setdiff(0:(mc_repeats - 1), completed_seeds)
  
  cat("\n================ RESUME STATUS ================\n")
  cat("tag =", tag, "\n")
  cat("completed seeds =", length(completed_seeds), "\n")
  if (length(completed_seeds) > 0) {
    cat("completed seed list =", paste(completed_seeds, collapse = ", "), "\n")
  }
  cat("remaining seeds =", length(remaining_seeds), "\n")
  if (length(remaining_seeds) > 0) {
    cat("remaining seed list =", paste(remaining_seeds, collapse = ", "), "\n")
  }
  cat("===============================================\n\n")
  
  for (seed in remaining_seeds) {
    cat("\n====================================================\n")
    cat("scenario =", scenario_id, "| p =", p, "| seed =", seed, "\n")
    cat("====================================================\n")
    
    one <- run_one_seed_grf(
      scenario_id = scenario_id,
      p = p,
      train_n = train_n,
      seed = seed,
      tau = tau,
      input_dir = input_dir
    )
    
    seed_cache_path <- file.path(seed_cache_dir, paste0(tag, "_seed", seed, ".rds"))
    save_one_seed_cache(one, seed_cache_path)
    
    cat("seed =", seed,
        "| test comp =", sprintf("%.6f", one$comp_df$test_comp),
        "| cross =", sprintf("%.6f", one$comp_df$test_cross),
        "| saved =", seed_cache_path, "\n")
  }
  
  loaded_final <- load_completed_seed_results(
    seed_cache_dir = seed_cache_dir,
    tag = tag,
    mc_repeats = mc_repeats
  )
  
  per_tau_all <- loaded_final$per_tau_all
  comp_all    <- loaded_final$comp_all
  tuning_all  <- loaded_final$tuning_all
  
  if (is.null(per_tau_all) || is.null(comp_all) || is.null(tuning_all)) {
    stop("저장된 seed 결과가 없습니다.")
  }
  
  per_tau_all <- per_tau_all[order(per_tau_all$seed, per_tau_all$tau), ]
  comp_all    <- comp_all[order(comp_all$seed), ]
  tuning_all  <- tuning_all[order(tuning_all$seed, tuning_all$candidate_id), ]
  
  summary <- summarize_mc_results(
    per_tau_all = per_tau_all,
    comp_all = comp_all,
    tau = tau,
    model_name = model_name
  )
  
  raw_csv_per_tau <- file.path(out_root, "raw", paste0(tag, "_per_tau_raw.csv"))
  raw_csv_comp    <- file.path(out_root, "raw", paste0(tag, "_comp_raw.csv"))
  raw_csv_tuning  <- file.path(out_root, "raw", paste0(tag, "_tuning_raw.csv"))
  out_xlsx        <- file.path(out_root, "excel", paste0(tag, ".xlsx"))
  out_png         <- file.path(out_root, "plots", paste0(tag, "_tau_curve.png"))
  out_rds         <- file.path(out_root, "raw", paste0(tag, "_raw_result.rds"))
  
  write.csv(per_tau_all, raw_csv_per_tau, row.names = FALSE)
  write.csv(comp_all, raw_csv_comp, row.names = FALSE)
  write.csv(tuning_all, raw_csv_tuning, row.names = FALSE)
  
  save_summary_excel(summary, out_xlsx, model_name = model_name)
  plot_tau_curve(summary, out_png, tag = tag, model_name = model_name)
  save_raw_results_rds(per_tau_all, comp_all, tuning_all, tau, out_rds, model_name = model_name)
  
  cat("\n================ FINAL SUMMARY ================\n")
  cat("tag =", tag, "\n")
  cat("n_mc =", summary$n_mc, "\n")
  cat("comp_mean =", summary$comp_mean, "\n")
  cat("comp_SE =", summary$comp_se, "\n")
  cat("cross_mean =", summary$cross_mean, "\n")
  cat("cross_SE =", summary$cross_se, "\n")
  cat("[Saved per_tau CSV] ", raw_csv_per_tau, "\n", sep = "")
  cat("[Saved comp CSV]    ", raw_csv_comp, "\n", sep = "")
  cat("[Saved tuning CSV]  ", raw_csv_tuning, "\n", sep = "")
  cat("[Saved Excel]       ", out_xlsx, "\n", sep = "")
  cat("[Saved Plot]        ", out_png, "\n", sep = "")
  cat("[Saved RDS]         ", out_rds, "\n", sep = "")
  cat("===============================================\n")
  
  invisible(list(
    summary = summary,
    per_tau_all = per_tau_all,
    comp_all = comp_all,
    tuning_all = tuning_all
  ))
}

# 3. 실행 --------------------------------------------------------------

res_grf_mc <- run_mc_grf(
  scenario_id = scenario_id,
  p = p,
  train_n = train_n,
  mc_repeats = mc_repeats,
  tau = tau,
  input_dir = input_dir,
  out_root = out_root,
  model_name = "grf"
)