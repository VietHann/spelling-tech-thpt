# Makefile for LaTeX report compilation

# Variables
MAIN = report
TEX = pdflatex
BIB = bibtex
VIEWER = evince  # Change to 'open' on macOS, 'start' on Windows

# Default target
all: $(MAIN).pdf

# Compile PDF
$(MAIN).pdf: $(MAIN).tex
	@echo "Compiling LaTeX document..."
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex
	@echo "Done! Output: $(MAIN).pdf"

# Compile with bibliography
full: $(MAIN).tex
	@echo "Full compilation with bibliography..."
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex
	@echo "Done! Output: $(MAIN).pdf"

# View PDF
view: $(MAIN).pdf
	$(VIEWER) $(MAIN).pdf &

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).lof $(MAIN).lot
	rm -f *.aux *.log *.out *.toc *.bbl *.blg
	@echo "Clean complete!"

# Clean all including PDF
cleanall: clean
	@echo "Removing PDF..."
	rm -f $(MAIN).pdf
	@echo "All clean!"

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Compile PDF (quick)"
	@echo "  make full     - Full compilation with bibliography"
	@echo "  make view     - View PDF"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make help     - Show this help"

.PHONY: all full view clean cleanall help

