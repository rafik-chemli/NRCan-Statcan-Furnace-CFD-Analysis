# Makefile for CFD figure generation

.PHONY: all clean figures

# Default target
all: figures

# Configuration
OUTPUT_DIR = figures
DATA_FILE = final_states_with_clusters.csv
ALGORITHM = Spectral
CLUSTERS = 100
FEATURES = primary_flow pressure density particle_feed diameter
LABELS = "Primary gas flow (kg/s)" "Pressure (bar)" "Particle density (kg/m3)" "Particle feed rate (kg/s)" "Particle diameter (m)"

# Create figures
figures:
	@echo "Generating publication-quality figures for CFD analysis..."
	@python3 Clustering_figures.py \
		--data $(DATA_FILE) \
		--output $(OUTPUT_DIR) \
		--n_clusters $(CLUSTERS) \
		--clustering $(ALGORITHM) \
		--features $(FEATURES) \
		--labels $(LABELS)
	@echo "Figure generation complete."
	@echo "Figures saved to:"
	@for n in $(CLUSTERS); do echo "  - $(OUTPUT_DIR)/$${n}_clusters/"; done
	@echo "Available figures:"
	@ls -la $(OUTPUT_DIR)/*_clusters/*.png
	@echo "Caption files:"
	@ls -la $(OUTPUT_DIR)/*_clusters/figure_captions.txt

# Clean generated files
clean:
	@echo "Cleaning up generated figures..."
	@rm -rf $(OUTPUT_DIR)
	@echo "Done."