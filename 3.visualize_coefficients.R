# Gregory Way 2018
# Multiple Myeloma Classifier
# 3.visualize_coefficients.R
#
# Observes differences in coefficients across the multiclass classifier classes
#
# Usage: Run in command line
#
#     Rscript --vanilla visualize_coefficients.R
#
# Output:
# Descriptive Coefficient Plot

library(dplyr)
library(ggplot2)
library(ggrepel)

# Load ENSEMBL to Symbol dictionary
gene_file <- file.path('data', 'raw', 'gprofiler_results_1002952837509.xlsx')
gene_dict_df <- readxl::read_excel(gene_file, col_names = FALSE)
colnames(gene_dict_df) <- c('num', 'GENE_ID', 'co', 'SYMBOL', 'ALT', 'a', 'b')
gene_dict_df <- gene_dict_df[!duplicated(gene_dict_df$GENE_ID), ]

# Load Classifier Coefficients
coef_file <- file.path('results', 'classifier', 'classifier_coefficients.tsv')
coef_df <- readr::read_tsv(coef_file) %>%
  dplyr::left_join(gene_dict_df, by = 'GENE_ID')

# Define cutoffs to label top 10 genes (by absolute value) for each classifier
# Note that dplyr still has some serious issues with writing functions
n <- 5

top_kras <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, KRAS) %>%
  dplyr::select(KRAS) %>%
  dplyr::arrange(desc(KRAS)) %>%
  dplyr::filter(row_number() == n()))

bot_kras <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, desc(KRAS)) %>%
  dplyr::select(KRAS) %>%
  dplyr::arrange(KRAS) %>%
  dplyr::filter(row_number()==n()))

top_wt <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, wildtype) %>%
  dplyr::select(wildtype) %>%
  dplyr::arrange(desc(wildtype)) %>%
  dplyr::filter(row_number() == n()))

bot_wt <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, desc(wildtype)) %>%
  dplyr::select(wildtype) %>%
  dplyr::arrange(wildtype) %>%
  dplyr::filter(row_number()==n()))

top_nras <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, NRAS) %>%
  dplyr::select(NRAS) %>%
  dplyr::arrange(desc(NRAS)) %>%
  dplyr::filter(row_number() == n()))

bot_nras <- as.numeric(coef_df %>%
  dplyr::top_n(n = n, desc(NRAS)) %>%
  dplyr::select(NRAS) %>%
  dplyr::arrange(NRAS) %>%
  dplyr::filter(row_number()==n()))

# Plot gene coefficients scatter
p <- ggplot2::ggplot(coef_df,
                     aes(x = KRAS, y = NRAS, color = wildtype)) +
  geom_point(alpha = 0.8, size = 0.1) +
  scale_color_gradient2('Wildtype\nGene Weight',
                        low = "blue",
                        mid = "grey",
                        high = "red") +
  xlab("KRAS - Gene Weight") +
  ylab("NRAS - Gene Weight") +
  geom_text_repel(data = subset(coef_df,
                                (wildtype >= top_wt | wildtype <= bot_wt) |
                                  (KRAS >= top_kras | KRAS <= bot_kras) |
                                  (NRAS >= top_nras | NRAS <= bot_nras)
  ),
  arrow = arrow(length = unit(0.02, 'npc')),
  segment.size = 0.3,
  segment.alpha = 0.6,
  box.padding = 0.17,
  point.padding = 0.1,
  size = 1.8,
  fontface = 'italic',
  aes(x = KRAS, y = NRAS, label = SYMBOL)) +
  theme_bw() +
  theme(axis.text = element_text(size = rel(0.5)),
        axis.title = element_text(size = rel(0.6)),
        axis.title.y = element_text(margin = 
                                      margin(t = 0, r = 0, b = 0, l = 0)),
        axis.title.x = element_text(margin =
                                      margin(t = 3, r = 0, b = 0, l = 0)),
        legend.text = element_text(size = rel(0.4)),
        legend.title = element_text(size = rel(0.5)),
        legend.key = element_rect(size = 0.2),
        legend.position = 'right',
        legend.key.size = unit(0.4, 'lines'),
        legend.margin = margin(l = -0.3, unit = 'cm'))

fig_file <- file.path('figures', 'classifier_coefficients_scatter.pdf')
ggplot2::ggsave(fig_file, plot = p, dpi = 600, width = 4, height = 3)
