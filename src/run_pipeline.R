suppressPackageStartupMessages({
  library(imager)
  library(class)
  library(glmnet)
})

source("src/utils.R")

cat("Project started...\n")

# -----------------------------
# Setup
# -----------------------------
image_dir <- "data"
image_height <- 64
image_width  <- 64
k_feat <- 76
set.seed(123)

# -----------------------------
# Load and preprocess images
# -----------------------------
image_files <- list.files(
  image_dir,
  pattern = "\\.(jpg|jpeg|png|bmp)$",
  ignore.case = TRUE,
  full.names = TRUE
)

if (length(image_files) == 0) {
  stop("No images found in 'data/'. Put images in data/ and try again.")
}

cat("Number of images:", length(image_files), "\n")

image_matrix_list <- lapply(image_files, read_to_vector, H = image_height, W = image_width)
x <- do.call(rbind, image_matrix_list)

# -----------------------------
# Labels
# -----------------------------
file_names <- basename(image_files)
labels <- ifelse(grepl("^c", file_names, ignore.case = TRUE), "Chelsea", "ManUtd")
labels <- factor(labels)

cat("Class distribution:\n")
print(table(labels))

# -----------------------------
# PCA
# -----------------------------
pca_full <- prcomp(x, center = TRUE, scale. = FALSE)
var_ratio <- (pca_full$sdev^2) / sum(pca_full$sdev^2)
cumvar <- cumsum(var_ratio)
cat(sprintf("Explained variance at k=%d: %.3f\n", k_feat, cumvar[k_feat]))

Z_all <- pca_full$x[, 1:k_feat, drop = FALSE]

# -----------------------------
# Train/Test split
# -----------------------------
train_idx <- sample(nrow(Z_all), 0.8 * nrow(Z_all))
X_train <- Z_all[train_idx, , drop = FALSE]
X_test  <- Z_all[-train_idx, , drop = FALSE]
y_train <- labels[train_idx]
y_test  <- labels[-train_idx]

positive_class <- "ManUtd"

# -----------------------------
# Part 2: PCA-only models
# -----------------------------
Kfold  <- 5
k_grid <- c(1, 3, 5, 7, 9, 11)

fold_id <- sample(rep(1:Kfold, length.out = length(y_train)))

cv_f1_knn <- sapply(k_grid, function(k) {
  f1_per_fold <- sapply(1:Kfold, function(fold) {
    val <- fold_id == fold
    tr  <- !val

    pred_val <- knn(
      train = X_train[tr, , drop = FALSE],
      test  = X_train[val, , drop = FALSE],
      cl    = y_train[tr],
      k     = k
    )

    metrics_binary(y_train[val], pred_val, positive = positive_class)$f1
  })
  mean(f1_per_fold, na.rm = TRUE)
})

best_k <- k_grid[which.max(cv_f1_knn)]
cat("Best KNN k (PCA-only):", best_k, "\n")

knn_test_pred <- knn(train = X_train, test = X_test, cl = y_train, k = best_k)
metrics_knn_test <- metrics_binary(y_test, knn_test_pred, positive = positive_class)

# Logistic (no reg) - train full, test
df_train <- data.frame(X_train)
df_train$label <- y_train
glm_fit_full <- glm(label ~ ., data = df_train, family = binomial)

df_test <- data.frame(X_test)
prob_test_glm <- predict(glm_fit_full, newdata = df_test, type = "response")
pred_test_glm <- factor(ifelse(prob_test_glm >= 0.5, positive_class, "Chelsea"),
                        levels = levels(y_train))
metrics_glm_test <- metrics_binary(y_test, pred_test_glm, positive = positive_class)

# Logistic L1/L2 (glmnet)
y_train_bin <- ifelse(y_train == positive_class, 1, 0)
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

fit_glmnet_model <- function(alpha_value) {
  set.seed(123)
  cv_fit <- cv.glmnet(
    x = X_train_mat,
    y = y_train_bin,
    family = "binomial",
    alpha = alpha_value,
    nfolds = 5
  )

  prob_test <- predict(cv_fit, newx = X_test_mat, s = "lambda.min", type = "response")
  pred_test <- factor(ifelse(prob_test >= 0.5, positive_class, "Chelsea"),
                      levels = levels(y_train))

  list(metrics = metrics_binary(y_test, pred_test, positive = positive_class),
       lambda_min = cv_fit$lambda.min)
}

l1_out <- fit_glmnet_model(alpha_value = 1)
l2_out <- fit_glmnet_model(alpha_value = 0)

metrics_l1_test <- l1_out$metrics
metrics_l2_test <- l2_out$metrics

cat("Best lambda (L1):", l1_out$lambda_min, "\n")
cat("Best lambda (L2):", l2_out$lambda_min, "\n")

results_table <- rbind(
  metrics_to_row(metrics_knn_test, "KNN (PCA-only)"),
  metrics_to_row(metrics_glm_test, "Logistic (no reg, PCA-only)"),
  metrics_to_row(metrics_l1_test,  "Logistic L1 (PCA-only)"),
  metrics_to_row(metrics_l2_test,  "Logistic L2 (PCA-only)")
)

cat("\n=== PCA-only Results ===\n")
print(results_table)

# -----------------------------
# Part 3: Nonlinear features
# -----------------------------
Z_nl_all <- make_nonlinear_features(Z_all)
Xnl_train <- Z_nl_all[train_idx, , drop = FALSE]
Xnl_test  <- Z_nl_all[-train_idx, , drop = FALSE]

# KNN nonlinear
fold_id2 <- sample(rep(1:Kfold, length.out = length(y_train)))

cv_f1_knn_nl <- sapply(k_grid, function(k) {
  f1_per_fold <- sapply(1:Kfold, function(fold) {
    val <- fold_id2 == fold
    tr  <- !val

    pred_val <- knn(
      train = Xnl_train[tr, , drop = FALSE],
      test  = Xnl_train[val, , drop = FALSE],
      cl    = y_train[tr],
      k     = k
    )

    metrics_binary(y_train[val], pred_val, positive = positive_class)$f1
  })
  mean(f1_per_fold, na.rm = TRUE)
})

best_k_nl <- k_grid[which.max(cv_f1_knn_nl)]
cat("Best KNN k (Nonlinear):", best_k_nl, "\n")

knn_nl_test_pred <- knn(train = Xnl_train, test = Xnl_test, cl = y_train, k = best_k_nl)
metrics_knn_nl_test <- metrics_binary(y_test, knn_nl_test_pred, positive = positive_class)

# Logistic nonlinear (no reg)
df_train_nl <- data.frame(Xnl_train, label = y_train)
glm_nl_fit_full <- glm(label ~ ., data = df_train_nl, family = binomial)

df_test_nl <- data.frame(Xnl_test)
prob_test_glm_nl <- predict(glm_nl_fit_full, newdata = df_test_nl, type = "response")
pred_test_glm_nl <- factor(ifelse(prob_test_glm_nl >= 0.5, positive_class, "Chelsea"),
                           levels = levels(y_train))
metrics_glm_nl_test <- metrics_binary(y_test, pred_test_glm_nl, positive = positive_class)

# glmnet nonlinear
Xnl_train_mat <- as.matrix(Xnl_train)
Xnl_test_mat  <- as.matrix(Xnl_test)

fit_glmnet_nl <- function(alpha_value) {
  set.seed(123)
  cv_fit <- cv.glmnet(
    x = Xnl_train_mat,
    y = y_train_bin,
    family = "binomial",
    alpha = alpha_value,
    nfolds = 5
  )

  prob_test <- predict(cv_fit, newx = Xnl_test_mat, s = "lambda.min", type = "response")
  pred_test <- factor(ifelse(prob_test >= 0.5, positive_class, "Chelsea"),
                      levels = levels(y_train))

  list(metrics = metrics_binary(y_test, pred_test, positive = positive_class),
       lambda_min = cv_fit$lambda.min)
}

l1_nl_out <- fit_glmnet_nl(alpha_value = 1)
l2_nl_out <- fit_glmnet_nl(alpha_value = 0)

results_table_nl <- rbind(
  metrics_to_row(metrics_knn_nl_test, "KNN (nonlinear)"),
  metrics_to_row(metrics_glm_nl_test, "Logistic (nonlinear)"),
  metrics_to_row(l1_nl_out$metrics,   "Logistic L1 (nonlinear)"),
  metrics_to_row(l2_nl_out$metrics,   "Logistic L2 (nonlinear)")
)

cat("\n=== Nonlinear Results ===\n")
print(results_table_nl)

# -----------------------------
# Save results
# -----------------------------
dir.create("reports", showWarnings = FALSE)
write.csv(results_table,    "reports/results_pca_only.csv", row.names = FALSE)
write.csv(results_table_nl, "reports/results_nonlinear.csv", row.names = FALSE)

cat("\nSaved:\n- reports/results_pca_only.csv\n- reports/results_nonlinear.csv\n")
cat("Done.\n")
