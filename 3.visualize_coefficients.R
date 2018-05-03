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

# Plot gene coefficients scatter
p <- ggplot2::ggplot(coef_df,
                aes(x = wildtype, y = KRAS, color = NRAS)) +
  geom_point(alpha = 0.8, size = 0.1) +
  scale_color_gradient2('NRAS\nGene Weight',
                        low = "blue",
                        mid = "grey",
                        high = "red") +
  xlab("Wildtype - Gene Weight") +
  ylab("KRAS - Gene Weight") +
  geom_text_repel(data = subset(coef_df,
                                (wildtype > 7 | wildtype < -8) |
                                  (KRAS > 9 | KRAS < -8.7) |
                                  (NRAS > 7.8 | NRAS < -8.6)
                                ),
                  arrow = arrow(length = unit(0.02, 'npc')),
                  segment.size = 0.3,
                  segment.alpha = 0.6,
                  box.padding = 0.17,
                  point.padding = 0.1,
                  size = 1.8,
                  fontface = 'italic',
                  aes(x = wildtype, y = KRAS, label = SYMBOL)) +
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
