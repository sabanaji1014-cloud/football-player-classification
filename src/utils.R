# src/utils.R

read_to_vector <- function(path, H = 64, W = 64){
  im <- imager::load.image(path)
  im <- imager::grayscale(im)
  im <- imager::resize(im, size_x = W, size_y = H)
  as.numeric(im)
}

reconstruct_with_k <- function(pca_obj, k){
  scores_k <- pca_obj$x[, 1:k, drop = FALSE]
  rotation_k <- pca_obj$rotation[, 1:k, drop = FALSE]
  x_hat_centered <- scores_k %*% t(rotation_k)
  sweep(x_hat_centered, 2, pca_obj$center, "+")
}

show_image <- function(vec, W = 64, H = 64, title = "") {
  im <- imager::as.cimg(vec, x = W, y = H, cc = 1)
  plot(im, axes = FALSE, main = title)
}

metrics_binary <- function(y_true, y_pred, positive = "ManUtd") {
  y_true <- factor(y_true)
  y_pred <- factor(y_pred, levels = levels(y_true))

  cm <- table(y_true, y_pred)

  TP <- cm[positive, positive]
  FP <- sum(cm[, positive]) - TP
  FN <- sum(cm[positive, ]) - TP
  TN <- sum(cm) - TP - FP - FN

  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1        <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0,
                      NA, 2 * precision * recall / (precision + recall))
  accuracy  <- (TP + TN) / sum(cm)

  list(
    precision = precision,
    recall    = recall,
    f1        = f1,
    accuracy  = accuracy,
    confusion = cm
  )
}

metrics_to_row <- function(m, name) {
  data.frame(
    Model     = name,
    Accuracy  = m$accuracy,
    Precision = m$precision,
    Recall    = m$recall,
    F1        = m$f1,
    stringsAsFactors = FALSE
  )
}

make_nonlinear_features <- function(Z, num_interact = 5){
  Z <- as.data.frame(Z)
  p <- ncol(Z)
  names(Z) <- paste0("PC", seq_len(p))

  base <- Z

  sq <- base^2
  names(sq) <- paste0(names(base), "_sq")

  lg <- log(abs(base) + 1)
  names(lg) <- paste0(names(base), "_log")

  m <- min(num_interact, p)
  pairs <- utils::combn(m, 2)

  inter <- sapply(seq_len(ncol(pairs)), function(k){
    i <- pairs[1,k]
    j <- pairs[2,k]
    base[[i]] * base[[j]]
  })

  inter <- as.data.frame(inter)
  names(inter) <- paste0("PC", pairs[1, ], "_x_PC", pairs[2, ])

  cbind(base, sq, lg, inter)
}