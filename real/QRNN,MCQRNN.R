library(qrnn)
library(jsonlite)
library(openxlsx)

# ============================================================
# 설정
# ============================================================
setwd("/home/hayn08/0505quantile")

shared_root <- "/home/hayn08/0505quantile/realdata_shared_randomsplit_repeat100"
out_root   <- "/home/hayn08/0505quantile/0603_real_qrnn_mcqrnn_result"

dataset_name <- "airfoil"
model_name <- "qrnn"   # "qrnn" or "mcqrnn"

tau <- c(0.1, 0.3, 0.5, 0.7, 0.9)

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 유틸
# ============================================================
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

  pred_mat <- as.matrix(pred_mat)

  if (ncol(pred_mat) < 2) {
    return(0)
  }

  mean(
    pred_mat[, -ncol(pred_mat), drop = FALSE] >
      pred_mat[, -1, drop = FALSE]
  ) * 100
}

mean_std_se <- function(x) {

  x <- as.numeric(x)

  list(
    mean = mean(x),
    std = sd(x),
    se = sd(x) / sqrt(length(x))
  )
}

save_summary_excel <- function(summary, out_xlsx, model_name) {

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

# ============================================================
# 데이터 로드
# ============================================================
load_saved_dataset_random <- function(dataset_name, shared_root) {

  dataset_dir <- file.path(shared_root, dataset_name)

  df <- read.csv(
    file.path(dataset_dir,
              paste0(dataset_name, "_full_data.csv")),
    check.names = FALSE
  )

  split_info <- fromJSON(
    file.path(dataset_dir,
              paste0(dataset_name, "_splits_random.json")),
    simplifyVector = FALSE
  )

  feature_cols <- setdiff(colnames(df), c("row_id", "y"))

  list(
    df = df,
    feature_cols = feature_cols,
    split_info = split_info
  )
}

# ============================================================
# 전처리
# ============================================================
scale_by_inner_train <- function(
    X_inner_train,
    X_inner_valid,
    X_train,
    X_test
) {

  Xtr_s <- scale(X_inner_train)

  center <- attr(Xtr_s, "scaled:center")
  scalev <- attr(Xtr_s, "scaled:scale")

  scalev[scalev == 0] <- 1

  list(
    X_inner_train = scale(
      X_inner_train,
      center = center,
      scale = scalev
    ),

    X_inner_valid = scale(
      X_inner_valid,
      center = center,
      scale = scalev
    ),

    X_train = scale(
      X_train,
      center = center,
      scale = scalev
    ),

    X_test = scale(
      X_test,
      center = center,
      scale = scalev
    )
  )
}

remove_zero_var <- function(
    X_inner_train,
    X_inner_valid,
    X_train,
    X_test
) {

  keep <- apply(
    X_inner_train,
    2,
    function(z) var(z, na.rm = TRUE) > 0
  )

  list(
    X_inner_train = X_inner_train[, keep, drop = FALSE],
    X_inner_valid = X_inner_valid[, keep, drop = FALSE],
    X_train = X_train[, keep, drop = FALSE],
    X_test = X_test[, keep, drop = FALSE]
  )
}

# ============================================================
# 데이터 split 준비
# ============================================================
prepare_split_data <- function(df, feature_cols, split_one) {

  inner_train_ids <- unlist(split_one[["inner_train_row_id"]])
  inner_valid_ids <- unlist(split_one[["inner_valid_row_id"]])

  train_ids <- unlist(split_one[["train_row_id"]])
  test_ids <- unlist(split_one[["test_row_id"]])

  df_inner_train <- df[df$row_id %in% inner_train_ids, ]
  df_inner_valid <- df[df$row_id %in% inner_valid_ids, ]

  df_train <- df[df$row_id %in% train_ids, ]
  df_test <- df[df$row_id %in% test_ids, ]

  X_inner_train <- as.matrix(
    df_inner_train[, feature_cols, drop = FALSE]
  )

  X_inner_valid <- as.matrix(
    df_inner_valid[, feature_cols, drop = FALSE]
  )

  X_train <- as.matrix(
    df_train[, feature_cols, drop = FALSE]
  )

  X_test <- as.matrix(
    df_test[, feature_cols, drop = FALSE]
  )

  scaled <- scale_by_inner_train(
    X_inner_train,
    X_inner_valid,
    X_train,
    X_test
  )

  filtered <- remove_zero_var(
    scaled$X_inner_train,
    scaled$X_inner_valid,
    scaled$X_train,
    scaled$X_test
  )

  list(
    X_inner_train = filtered$X_inner_train,
    y_inner_train = df_inner_train$y,

    X_inner_valid = filtered$X_inner_valid,
    y_inner_valid = df_inner_valid$y,

    X_train = filtered$X_train,
    y_train = df_train$y,

    X_test = filtered$X_test,
    y_test = df_test$y
  )
}

# ============================================================
# QRNN tuning
# ============================================================
tune_qrnn_once <- function(
    X_inner_train,
    y_inner_train,
    X_inner_valid,
    y_inner_valid,
    tau,
    repeat_id,
    seed = 100
) {

  # ④ MCQRNN 그리드: n.hidden 범위 확장
    grid <- expand.grid(
      n.hidden  = c(16, 32, 64),  # 64 추가
      penalty   = c(0, 1e-4),
      iter.max  = c(1000)          # 500 → 1000
    )

  y_inner_train <- matrix(y_inner_train, ncol = 1)

  tuning_rows <- list()

  best_val <- Inf
  best_idx <- NA

  for (i in seq_len(nrow(grid))) {

    params <- grid[i, ]

    pred_valid <- matrix(
      NA,
      nrow = nrow(X_inner_valid),
      ncol = length(tau)
    )

    for (j in seq_along(tau)) {

      set.seed(seed + repeat_id * 1000 + i * 100 + j)

      fit <- qrnn.fit(
        x         = X_inner_train,
        y         = y_inner_train,
        tau       = tau[j],
        n.hidden  = params$n.hidden,
        n.trials  = 3,              # ② 1 → 3 이상으로 (최종 fit은 5)
        iter.max  = 1000,           # ③ 500 → 1000
        penalty   = params$penalty,
        trace     = FALSE
        )

      pred_valid[, j] <- as.numeric(
        qrnn.predict(X_inner_valid, fit)
      )
    }

    val_res <- composite_pinball(
      y_inner_valid,
      pred_valid,
      tau
    )

    tuning_rows[[i]] <- data.frame(
      repeat_id = repeat_id,
      candidate_id = i,
      n.hidden = params$n.hidden,
      penalty = params$penalty,
      iter.max = params$iter.max,
      valid_comp = val_res$comp
    )

    if (val_res$comp < best_val) {
      best_val <- val_res$comp
      best_idx <- i
    }
  }

  list(
    best_params = as.list(grid[best_idx, ]),
    best_valid_comp = best_val,
    tuning_df = do.call(rbind, tuning_rows)
  )
}

# ============================================================
# MCQRNN tuning
# ============================================================
tune_mcqrnn_once <- function(
    X_inner_train,
    y_inner_train,
    X_inner_valid,
    y_inner_valid,
    tau,
    repeat_id,
    seed = 100
) {

    # tune_mcqrnn_once 안 — 수정 필요
    grid <- expand.grid(
      n.hidden  = c(16, 32, 64),   # 64 추가
      n.hidden2 = c(NA, 16),
      penalty   = c(0, 1e-4),
      iter.max  = c(1000)           # 500 → 1000
    )

  y_inner_train <- matrix(y_inner_train, ncol = 1)

  tuning_rows <- list()

  best_val <- Inf
  best_params <- NULL

  for (i in seq_len(nrow(grid))) {

    params <- grid[i, ]

    nh2 <- if (is.na(params$n.hidden2)) {
      NULL
    } else {
      params$n.hidden2
    }

    set.seed(seed + repeat_id * 1000 + i)

    # MCQRNN tuning fit
    fit <- tryCatch({
      mcqrnn.fit(
        x         = X_inner_train,
        y         = y_inner_train,
        tau       = tau,
        n.hidden  = params$n.hidden,
        n.hidden2 = nh2,
        n.trials  = 3,             # ② 추가
        iter.max  = params$iter.max,
        penalty   = params$penalty,
        trace     = FALSE
      )
    }, error = function(e) NULL)

    if (is.null(fit)) {
      next
    }

    pred_valid <- as.matrix(
      mcqrnn.predict(X_inner_valid, fit, tau = tau)
    )

    val_res <- composite_pinball(
      y_inner_valid,
      pred_valid,
      tau
    )

    tuning_rows[[length(tuning_rows) + 1]] <- data.frame(
      repeat_id = repeat_id,
      n.hidden = params$n.hidden,
      n.hidden2 = ifelse(is.null(nh2), NA, nh2),
      penalty = params$penalty,
      iter.max = params$iter.max,
      valid_comp = val_res$comp
    )

    if (val_res$comp < best_val) {
      best_val <- val_res$comp
      best_params <- as.list(params)
    }
  }

  list(
    best_params = best_params,
    best_valid_comp = best_val,
    tuning_df = do.call(rbind, tuning_rows)
  )
}

# ============================================================
# repeat 1개 실행
# ============================================================
run_one_eval <- function(
    df,
    feature_cols,
    split_one,
    tau,
    model_name,
    seed = 100
) {

  repeat_id <- split_one[["repeat"]]

  dat <- prepare_split_data(
    df,
    feature_cols,
    split_one
  )

  if (model_name == "qrnn") {

    tune_res <- tune_qrnn_once(
      dat$X_inner_train,
      dat$y_inner_train,
      dat$X_inner_valid,
      dat$y_inner_valid,
      tau,
      repeat_id,
      seed
    )

  } else {

    tune_res <- tune_mcqrnn_once(
      dat$X_inner_train,
      dat$y_inner_train,
      dat$X_inner_valid,
      dat$y_inner_valid,
      tau,
      repeat_id,
      seed
    )
  }

  best_params <- tune_res$best_params

  # ========================================================
  # final fit
  # ========================================================
  pred_test <- matrix(
    NA,
    nrow = nrow(dat$X_test),
    ncol = length(tau)
  )

  if (model_name == "qrnn") {

    y_train_mat <- matrix(dat$y_train, ncol = 1)

    for (j in seq_along(tau)) {

      set.seed(seed + 10000 + repeat_id * 1000 + j)

      # QRNN 최종 fit
      fit <- qrnn.fit(
        x         = dat$X_train,
        y         = y_train_mat,
        tau       = tau[j],
        n.hidden  = best_params$n.hidden,
        n.trials  = 5,              # 최종 fit은 5
        iter.max  = 1000,
        penalty   = best_params$penalty,
        trace     = FALSE
      ) 
      pred_test[, j] <- as.numeric(
        qrnn.predict(dat$X_test, fit)
      )
    }

  } else {

    y_train_mat <- matrix(dat$y_train, ncol = 1)

    nh2 <- if (is.na(best_params$n.hidden2)) {
      NULL
    } else {
      best_params$n.hidden2
    }

    set.seed(seed + 10000 + repeat_id * 1000)

    fit <- mcqrnn.fit(
      x = dat$X_train,
      y = y_train_mat,
      tau = tau,
      n.hidden = best_params$n.hidden,
      n.hidden2 = nh2,
      n.trials = 3,
      iter.max = best_params$iter.max,
      penalty = best_params$penalty,
      trace = FALSE
    )

    pred_test <- as.matrix(
      mcqrnn.predict(dat$X_test, fit, tau = tau)
    )
  }

  test_res <- composite_pinball(
    dat$y_test,
    pred_test,
    tau
  )

  cross <- crossing_percentage(pred_test)

  per_tau_df <- data.frame(
    repeat_id = repeat_id,
    tau = tau,
    test_pinball = test_res$per_tau
  )

  comp_df <- data.frame(
    repeat_id = repeat_id,
    test_comp = test_res$comp,
    crossing = cross,
    best_valid_comp = tune_res$best_valid_comp
  )

  list(
    per_tau_df = per_tau_df,
    comp_df = comp_df,
    tuning_df = tune_res$tuning_df
  )
}

# ============================================================
# summary
# ============================================================
summarize_results <- function(
    per_tau_all,
    comp_all,
    tau,
    model_name
) {

  mean_tau <- numeric(length(tau))
  se_tau <- numeric(length(tau))

  for (j in seq_along(tau)) {

    vals <- per_tau_all$test_pinball[
      per_tau_all$tau == tau[j]
    ]

    tmp <- mean_std_se(vals)

    mean_tau[j] <- tmp$mean
    se_tau[j] <- tmp$se
  }

  comp_stat <- mean_std_se(comp_all$test_comp)
  cross_stat <- mean_std_se(comp_all$crossing)

  list(
    model_name = model_name,
    tau = tau,
    n_eval = nrow(comp_all),

    mean_tau = mean_tau,
    se_tau = se_tau,

    comp_mean = comp_stat$mean,
    comp_se = comp_stat$se,

    cross_mean = cross_stat$mean,
    cross_se = cross_stat$se
  )
}

# ============================================================
# 메인 실행
# ============================================================
run_real_qrnn_mcqrnn <- function(
    dataset_name,
    model_name,
    tau,
    shared_root,
    out_root,
    seed = 100
) {

  loaded <- load_saved_dataset_random(
    dataset_name,
    shared_root
  )

  df <- loaded$df
  feature_cols <- loaded$feature_cols
  split_info <- loaded$split_info

  per_tau_list <- list()
  comp_list <- list()
  tuning_list <- list()

  for (i in seq_along(split_info$splits)) {

    split_one <- split_info$splits[[i]]

    repeat_id <- split_one[["repeat"]]

    cat("\n====================================\n")
    cat("dataset =", dataset_name,
        "| model =", model_name,
        "| repeat =", repeat_id, "\n")
    cat("====================================\n")

    one <- run_one_eval(
      df,
      feature_cols,
      split_one,
      tau,
      model_name,
      seed
    )

    per_tau_list[[i]] <- one$per_tau_df
    comp_list[[i]] <- one$comp_df
    tuning_list[[i]] <- one$tuning_df

    cat(
      "[DONE]",
      "repeat =", repeat_id,
      "| comp =", round(one$comp_df$test_comp, 6),
      "| cross =", round(one$comp_df$crossing, 6),
      "\n"
    )
  }

  per_tau_all <- do.call(rbind, per_tau_list)
  comp_all <- do.call(rbind, comp_list)
  tuning_all <- do.call(rbind, tuning_list)

  summary <- summarize_results(
    per_tau_all,
    comp_all,
    tau,
    model_name
  )

  dataset_out <- file.path(out_root, dataset_name)

  dir.create(
    file.path(dataset_out, "raw"),
    recursive = TRUE,
    showWarnings = FALSE
  )

  dir.create(
    file.path(dataset_out, "excel"),
    recursive = TRUE,
    showWarnings = FALSE
  )

  tag <- paste0(model_name, "_", dataset_name)

  write.csv(
    per_tau_all,
    file.path(dataset_out, "raw",
              paste0(tag, "_per_tau_raw.csv")),
    row.names = FALSE
  )

  write.csv(
    comp_all,
    file.path(dataset_out, "raw",
              paste0(tag, "_comp_raw.csv")),
    row.names = FALSE
  )

  write.csv(
    tuning_all,
    file.path(dataset_out, "raw",
              paste0(tag, "_tuning_raw.csv")),
    row.names = FALSE
  )

  save_summary_excel(
    summary,
    file.path(dataset_out, "excel",
              paste0(tag, ".xlsx")),
    model_name
  )

  cat("\n================ FINAL ================\n")
  cat("dataset =", dataset_name, "\n")
  cat("model =", model_name, "\n")
  cat("comp mean =", summary$comp_mean, "\n")
  cat("comp se =", summary$comp_se, "\n")
  cat("cross mean =", summary$cross_mean, "\n")
  cat("cross se =", summary$cross_se, "\n")
  cat("=======================================\n")
}

# ============================================================
# 실행
# ============================================================

run_real_qrnn_mcqrnn(
  dataset_name = dataset_name,
  model_name = model_name,
  tau = tau,
  shared_root = shared_root,
  out_root = out_root,
  seed = 100
)

# MCQRNN 실행 예시
# run_real_qrnn_mcqrnn(
#   dataset_name = dataset_name,
#   model_name = "mcqrnn",
#   tau = tau,
#   shared_root = shared_root,
#   out_root = out_root,
#   seed = 100
# )