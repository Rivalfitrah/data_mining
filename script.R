# ============================================================
# 1. INSTALASI (jalankan sekali saja)
# ============================================================
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("randomForest")
# install.packages("dplyr")
# install.packages("ggplot2")

# load library
library(tidyverse)
library(randomForest)
library(caret)
library(smotefamily)
library(ggplot2)

# load dataset
spotify <- read_csv("data/spotify_churn_dataset.csv", show_col_types = FALSE)

cat("6 Baris Data Pertama:\n")
print(head(spotify))

cat("\nStruktur Dataset Awal:\n")
glimpse(spotify)

cat("\nCek Data Hilang (NA):\n")
print(colSums(is.na(spotify)))

cat("\nCek Duplikasi:\n")
print(sum(duplicated(spotify)))

str(spotify)

# cek data kotor
spotify %>% 
  filter(avg_daily_minutes < 0) %>% 
  head()

sum(spotify$avg_daily_minutes < 0, na.rm = TRUE)

median_val <- median(spotify$avg_daily_minutes[spotify$avg_daily_minutes >= 0], na.rm = TRUE)

spotify <- spotify %>%
  mutate(avg_daily_minutes = ifelse(avg_daily_minutes < 0, median_val, avg_daily_minutes))

cat("\nCek kembali nilai negatif setelah diisi median:\n")
print(sum(spotify$avg_daily_minutes < 0))

# 3.2 Hapus user_id dan Siapkan Target
# user_id tidak dipakai sebagai fitur (dibuang)
spotify_clean <- spotify %>% 
  select(-user_id)

# 3.3 ONE-HOT ENCODING (Wajib sebelum SMOTE)
# Mengubah kategori (country, genre) menjadi kolom angka (dummy variables)
dummies_model <- dummyVars(churned ~ ., data = spotify_clean)
data_numeric <- predict(dummies_model, newdata = spotify_clean)
data_numeric <- as.data.frame(data_numeric)

# PERBAIKAN ERROR "Hip-Hop": membuat nama kolom aman
colnames(data_numeric) <- make.names(colnames(data_numeric))

# Gabungkan kembali target churned (masih dalam bentuk angka 0/1 untuk SMOTE)
data_numeric$churned <- spotify_clean$churned

#latih dataset
set.seed(123)
train_index <- createDataPartition(data_numeric$churned, p = 0.7, list = FALSE)

train_data <- data_numeric[train_index, ]
test_data  <- data_numeric[-train_index, ]

cat("\nJumlah kelas sebelum SMOTE:\n")
print(table(train_data$churned)) # Cek imbalance data

train_x <- train_data %>% select(-churned)
train_y <- train_data$churned

# Menggunakan dup_size = 0 untuk menyeimbangkan kelas secara penuh (dari Kode 2)
smote_output <- SMOTE(X = train_x, target = train_y, K = 5, dup_size = 0) 

# Ambil data hasil SMOTE
train_smote <- smote_output$data

# Rapikan nama kolom hasil SMOTE
colnames(train_smote)[ncol(train_smote)] <- "churned"
colnames(train_smote) <- make.names(colnames(train_smote)) # Jaga-jaga

# PENTING: Ubah 'churned' jadi FACTOR agar Random Forest melakukan KLASIFIKASI
train_smote$churned <- as.factor(train_smote$churned)
test_data$churned   <- as.factor(test_data$churned) # Data test juga harus factor

cat("\nJumlah kelas setelah SMOTE:\n")
print(table(train_smote$churned)) # Cek data setelah balance

rf_model <- randomForest(
  churned ~ .,
  data = train_smote,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

cat("\nHasil Model Random Forest:\n")
print(rf_model)


# Prediksi menggunakan test_data (yang sudah di-encoded)
pred <- predict(rf_model, test_data)

# Tampilkan Confusion Matrix
# Menggunakan mode = "prec_recall" (sudah diperbaiki case sensitivity-nya)
cat("\nHasil Evaluasi (Confusion Matrix & Metrik):")
confusionMatrix(pred, test_data$churned, positive = "1", mode = "prec_recall")

# Cek Variable Importance (Fitur yang paling berpengaruh)
varImpPlot(rf_model, main = "Variable Importance Random Forest")


# Menggunakan data 'spotify' awal (sebelum encoding)
p1 <- ggplot(spotify, aes(avg_daily_minutes)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribusi Rata-rata Menit Pemakaian Harian")

print(p1)


# Visualisasi 8.2: Perbandingan Kelas Sebelum dan Sesudah SMOTE
par(mfrow=c(1,2))
barplot(table(train_data$churned), main="Sebelum SMOTE (Imbalanced)", col="red", ylab="Jumlah User")
barplot(table(train_smote$churned), main="Setelah SMOTE (Balanced)", col="green", ylab="Jumlah User")

# Kembalikan layout plot ke default
par(mfrow=c(1,1))



# Pastikan library 'tidyverse' sudah terload (mengandung dplyr dan ggplot2)
# Pastikan library 'randomForest' dan 'caret' sudah terload

# =================================================================
# FUNGSI VISUALISASI HASIL RANDOM FOREST
# =================================================================

visualize_rf_results <- function(rf_model, cm_object) {
  
  # 1. VISUALISASI VARIABLE IMPORTANCE (Bar Chart)
  
  # Ekstrak data importance (Mean Decrease Gini)
  imp_data <- as.data.frame(importance(rf_model))
  
  # Gunakan MeanDecreaseGini (metrik default untuk klasifikasi)
  imp_data <- imp_data %>% 
    tibble::rownames_to_column(var = "Feature") %>% 
    rename(Importance = MeanDecreaseGini) %>% 
    arrange(desc(Importance))
  
  p1 <- ggplot(imp_data, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#1DB954") + # Warna hijau Spotify
    coord_flip() +
    labs(title = "1. Faktor Utama Prediksi Churn (Importance Score)",
         x = "Fitur",
         y = "Pentingnya (Mean Decrease Gini)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.y = element_text(size = 10))
  
  
  # 2. VISUALISASI CONFUSION MATRIX SEBAGAI HEATMAP
  
  # Ekstrak matrix table dari objek confusionMatrix
  cm_table <- as.table(cm_object)
  cm_df <- as.data.frame(cm_table)
  colnames(cm_df) <- c("Predicted", "Actual", "Count")
  
  # Hitung metrik kunci untuk subtitle
  accuracy_val <- round(cm_object$overall['Accuracy'], 3)
  recall_val <- round(cm_object$byClass['Recall'], 3)
  precision_val <- round(cm_object$byClass['Precision'], 3)
  
  # Label untuk heatmap (mengganti 0/1 dengan istilah bisnis)
  cm_df$Actual_Label <- factor(cm_df$Actual, levels = c("0", "1"), 
                               labels = c("Setia (0)", "Churn (1)"))
  cm_df$Predicted_Label <- factor(cm_df$Predicted, levels = c("0", "1"), 
                                  labels = c("Setia (0)", "Churn (1)"))
  
  p2 <- ggplot(cm_df, aes(x = Actual_Label, y = Predicted_Label, fill = Count)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "#004D7A") + # Skema warna gelap
    geom_text(aes(label = paste(Count, "\n(", round(Count/sum(cm_df$Count)*100, 1), "%)")), 
              vjust = 1, color = "black", size = 5) +
    labs(
      title = "2. Hasil Prediksi Model (Confusion Matrix Heatmap)",
      subtitle = paste0("Accuracy: ", accuracy_val, " | Recall: ", recall_val, 
                        " | Precision: ", precision_val),
      x = "Aktual (Referensi)",
      y = "Prediksi (Model)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "none",
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10))
  
  # Tampilkan plot secara terpisah
  print(p1)
  print(p2)
}


# 7. EVALUASI MODEL (REVISI)
pred <- predict(rf_model, test_data)

# Simpan hasil confusionMatrix ke variabel cm_result
cm_result <- confusionMatrix(pred, test_data$churned, positive = "1", mode = "prec_recall")

# Panggil fungsi visualisasi yang baru dibuat
visualize_rf_results(rf_model, cm_result)

