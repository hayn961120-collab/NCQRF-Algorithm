library(qrnn)
library(data.table)
library(dplyr)

# =========================================================
# 설정
# =========================================================

DATA_DIR <- "/home/hayn08/0505quantile/0414_grf_input_exact"
OUT_DIR  <- "/home/hayn08/0505quantile/0514_qrnn_cran_result_param"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TAU <- c(0.1, 0.3, 0.5, 0.7, 0.9)

SCENARIOS <- c(1, 2, 4)
P_LIST <- c(1, 20)

MC_REPEATS <- 100

# =========================================================
# pinball loss
# =========================================================

pinball_loss <- function(y, pred, tau) {
  e <- y - pred
  mean(pmax(tau * e, (tau - 1) * e))
}

composite_loss <- function(y, pred_mat, tau = TAU) {
  mean(sapply(seq_along(tau), function(j) {
    pinball_loss(y, pred_mat[, j], tau[j])
  }))
}

crossing_percentage <- function(pred_mat) {
  
  pred_mat <- as.matrix(pred_mat)
  
  mean(
    pred_mat[, -ncol(pred_mat), drop = FALSE] >
      pred_mat[, -1, drop = FALSE]
  ) * 100
}

# =========================================================
# 데이터 불러오기
# =========================================================

load_split_data <- function(scenario_id, p, seed) {
  
  base <- paste0(
    "s", scenario_id,
    "_p", p,
    "_n1000_seed", seed
  )
  
  train_path <- file.path(DATA_DIR, paste0(base, "_train.csv"))
  valid_path <- file.path(DATA_DIR, paste0(base, "_valid.csv"))
  test_path  <- file.path(DATA_DIR, paste0(base, "_test.csv"))
  
  train_df <- fread(train_path)
  valid_df <- fread(valid_path)
  test_df  <- fread(test_path)
  
  list(
    train = train_df,
    valid = valid_df,
    test  = test_df
  )
}

# =========================================================
# X / y 분리
# =========================================================

split_xy <- function(df) {
  
  y <- df$y
  
  X <- as.matrix(
    df[, !names(df) %in% c("y"), with = FALSE]
  )
  
  list(X = X, y = y)
}

# =========================================================
# scaling
# =========================================================

scale_by_train <- function(Xtr, Xva, Xfit, Xte) {
  
  Xtr_s <- scale(Xtr)
  
  center <- attr(Xtr_s, "scaled:center")
  scalev <- attr(Xtr_s, "scaled:scale")
  
  scalev[scalev == 0] <- 1
  
  list(
    Xtr = scale(Xtr, center = center, scale = scalev),
    Xva = scale(Xva, center = center, scale = scalev),
    Xfit = scale(Xfit, center = center, scale = scalev),
    Xte = scale(Xte, center = center, scale = scalev)
  )
}

# =========================================================
# zero variance 제거
# =========================================================

remove_zero_var <- function(Xtr, Xva, Xfit, Xte) {
  
  keep <- apply(Xtr, 2, function(z) {
    var(z) > 0
  })
  
  list(
    Xtr = Xtr[, keep, drop = FALSE],
    Xva = Xva[, keep, drop = FALSE],
    Xfit = Xfit[, keep, drop = FALSE],
    Xte = Xte[, keep, drop = FALSE]
  )
}

# =========================================================
# QRNN
# =========================================================

fit_predict_qrnn <- function(
    Xtr, ytr,
    Xva, yva,
    Xfit, yfit,
    Xte,
    seed
) {
  
  scaled <- scale_by_train(Xtr, Xva, Xfit, Xte)
  
  Xtr  <- scaled$Xtr
  Xva  <- scaled$Xva
  Xfit <- scaled$Xfit
  Xte  <- scaled$Xte
  
  filtered <- remove_zero_var(Xtr, Xva, Xfit, Xte)
  
  Xtr  <- filtered$Xtr
  Xva  <- filtered$Xva
  Xfit <- filtered$Xfit
  Xte  <- filtered$Xte
  
  ytr  <- matrix(ytr, ncol = 1)
  yfit <- matrix(yfit, ncol = 1)
  
  grid <- expand.grid(
    n.hidden = c(8, 16, 32),
    penalty  = c(0, 1e-4),
    iter.max = c(500)
  )
  
  best_loss <- Inf
  best_param <- NULL
  
  # -----------------------------
  # validation
  # -----------------------------
  
  for (g in seq_len(nrow(grid))) {
    
    pred_va <- matrix(
      NA,
      nrow = nrow(Xva),
      ncol = length(TAU)
    )
    
    for (j in seq_along(TAU)) {
      
      set.seed(seed + j)
      
      fit <- tryCatch({
        
        qrnn.fit(
          x = Xtr,
          y = ytr,
          tau = TAU[j],
          n.hidden = grid$n.hidden[g],
          n.trials = 1,
          iter.max = grid$iter.max[g],
          penalty = grid$penalty[g],
          trace = FALSE
        )
        
      }, error = function(e) NULL)
      
      if (is.null(fit)) next
      
      pred_va[, j] <- as.numeric(
        qrnn.predict(Xva, fit)
      )
    }
    
    if (anyNA(pred_va)) next
    
    comp <- composite_loss(yva, pred_va)
    
    if (comp < best_loss) {
      best_loss <- comp
      best_param <- grid[g, ]
    }
  }
  
  # -----------------------------
  # final fit
  # -----------------------------
  
  pred_test <- matrix(
    NA,
    nrow = nrow(Xte),
    ncol = length(TAU)
  )
  
  for (j in seq_along(TAU)) {
    
    set.seed(seed + j)
    
    fit <- qrnn.fit(
      x = Xfit,
      y = yfit,
      tau = TAU[j],
      n.hidden = best_param$n.hidden,
      n.trials = 1,
      iter.max = best_param$iter.max,
      penalty = best_param$penalty,
      trace = FALSE
    )
    
    pred_test[, j] <- as.numeric(
      qrnn.predict(Xte, fit)
    )
  }
  
  list(
    pred = pred_test,
    best_param = best_param,
    valid_loss = best_loss
  )
}

# =========================================================
# MCQRNN
# =========================================================

fit_predict_mcqrnn <- function(
    Xtr, ytr,
    Xva, yva,
    Xfit, yfit,
    Xte,
    seed
) {
  
  scaled <- scale_by_train(Xtr, Xva, Xfit, Xte)
  
  Xtr  <- scaled$Xtr
  Xva  <- scaled$Xva
  Xfit <- scaled$Xfit
  Xte  <- scaled$Xte
  
  filtered <- remove_zero_var(Xtr, Xva, Xfit, Xte)
  
  Xtr  <- filtered$Xtr
  Xva  <- filtered$Xva
  Xfit <- filtered$Xfit
  Xte  <- filtered$Xte
  
  ytr  <- matrix(ytr, ncol = 1)
  yfit <- matrix(yfit, ncol = 1)
  
  grid <- expand.grid(
    n.hidden = c(16, 32),
    penalty  = c(0, 1e-4),
    iter.max = c(200)
  )
  
  best_loss <- Inf
  best_param <- NULL
  
  # -----------------------------
  # validation
  # -----------------------------
  
  for (g in seq_len(nrow(grid))) {
    
    set.seed(seed + g)
    
    fit <- tryCatch({
      
      mcqrnn.fit(
        x = Xtr,
        y = ytr,
        tau = TAU,
        n.hidden = grid$n.hidden[g],
        n.trials = 1,
        iter.max = grid$iter.max[g],
        penalty = grid$penalty[g],
        trace = FALSE
      )
      
    }, error = function(e) NULL)
    
    if (is.null(fit)) next
    
    pred_va <- as.matrix(
      mcqrnn.predict(Xva, fit, tau = TAU)
    )
    
    comp <- composite_loss(yva, pred_va)
    
    if (comp < best_loss) {
      best_loss <- comp
      best_param <- grid[g, ]
    }
  }
  
  # -----------------------------
  # final fit
  # -----------------------------
  
  set.seed(seed)
  
  fit_final <- mcqrnn.fit(
    x = Xfit,
    y = yfit,
    tau = TAU,
    n.hidden = best_param$n.hidden,
    n.trials = 1,
    iter.max = best_param$iter.max,
    penalty = best_param$penalty,
    trace = FALSE
  )
  
  pred_test <- as.matrix(
    mcqrnn.predict(Xte, fit_final, tau = TAU)
  )
  
  list(
    pred = pred_test,
    best_param = best_param,
    valid_loss = best_loss
  )
}

# =========================================================
# 메인 실행
# =========================================================

ALL_RESULTS <- list()

for (scenario_id in SCENARIOS) {
  
  for (p in P_LIST) {
    
    for (model_name in c("mcqrnn")) {
      
      cat("\n====================================\n")
      cat("Scenario :", scenario_id, "\n")
      cat("p        :", p, "\n")
      cat("Model    :", model_name, "\n")
      cat("====================================\n")
      
      result_list <- list()
      
      for (seed in 0:(MC_REPEATS - 1)) {
        
        cat("[RUN] seed =", seed, "\n")
        
        dat <- load_split_data(
          scenario_id = scenario_id,
          p = p,
          seed = seed
        )
        
        tr <- split_xy(dat$train)
        va <- split_xy(dat$valid)
        te <- split_xy(dat$test)
        
        Xfit <- rbind(tr$X, va$X)
        yfit <- c(tr$y, va$y)
        
        if (model_name == "qrnn") {
          
          res <- fit_predict_qrnn(
            tr$X, tr$y,
            va$X, va$y,
            Xfit, yfit,
            te$X,
            seed
          )
          
        } else {
          
          res <- fit_predict_mcqrnn(
            tr$X, tr$y,
            va$X, va$y,
            Xfit, yfit,
            te$X,
            seed
          )
        }
        
        pred <- res$pred
        
        losses <- sapply(seq_along(TAU), function(j) {
          pinball_loss(te$y, pred[, j], TAU[j])
        })
        
        one_row <- data.frame(
          scenario = scenario_id,
          p = p,
          model = model_name,
          seed = seed,
          
          tau_0.1 = losses[1],
          tau_0.3 = losses[2],
          tau_0.5 = losses[3],
          tau_0.7 = losses[4],
          tau_0.9 = losses[5],
          
          composite = mean(losses),
          crossing = crossing_percentage(pred)
        )
        
        result_list[[length(result_list) + 1]] <- one_row
      }
      
      final_df <- bind_rows(result_list)
      
      out_path <- file.path(
        OUT_DIR,
        paste0(
          "scenario_", scenario_id,
          "_p", p,
          "_", model_name,
          "_raw.csv"
        )
      )
      
      fwrite(final_df, out_path)
      
      cat("[SAVED]", out_path, "\n")
    }
  }
}